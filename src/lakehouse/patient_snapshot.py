"""
src/lakehouse/patient_snapshot.py

Patient Snapshot API.

Core function: get_patient_snapshot(hadm_id) → dict + formatted text string.
Called by the retrieval layer for every T+E and T+E+K query.

Returns:
    {
        "hadm_id":        int,
        "subject_id":     int,
        "age":            int | None,
        "gender":         str | None,
        "diagnoses":      [{"icd_code": str, "description": str}, ...],
        "labs":           [{"label": str, "value": float, "unit": str, "flag": str}, ...],
        "vitals":         [{"label": str, "value": float, "unit": str}, ...],
        "medications":    [str, ...],
        "sparsity_score": float,
        "sparsity_bucket": str,          # "low" | "medium" | "high"
        "snapshot_text":  str,           # formatted string for LLM prompt
    }

Usage:
    from src.lakehouse.patient_snapshot import PatientSnapshot
    api = PatientSnapshot()
    snap = api.get(hadm_id=25120131)
    print(snap["snapshot_text"])

    # Or standalone test:
    python src/lakehouse/patient_snapshot.py --hadm-id 25120131
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

LAKE          = Path("data/lakehouse")
SPARSITY_FILE = LAKE / "sparsity.parquet"

MAX_LABS      = 10   # top labs to include in snapshot
MAX_VITALS    = 5    # top vitals to include
MAX_DIAGNOSES = 5    # top diagnoses
MAX_MEDS      = 8    # top medications

# Vital sign itemids (MIMIC-IV chartevents)
VITAL_ITEMIDS = {
    220045: "Heart Rate",
    220050: "Arterial BP Systolic",
    220051: "Arterial BP Diastolic",
    220052: "Arterial BP Mean",
    220210: "Respiratory Rate",
    223761: "Temperature (F)",
    220277: "SpO2",
    220739: "GCS Eye Opening",
    223900: "GCS Verbal Response",
    223901: "GCS Motor Response",
}


# ── DuckDB connection ─────────────────────────────────────────────────────────

class PatientSnapshot:
    """
    Manages DuckDB connection and Parquet views for patient snapshot retrieval.
    Reuse a single instance across multiple calls — connection is kept open.
    """

    def __init__(self, lake: Path = LAKE) -> None:
        self.lake = lake
        self._con: duckdb.DuckDBPyConnection | None = None
        self._schema: dict | None = None
        self._has_sparsity = False
        self._has_vitals   = False
        self._has_notes    = False
        self._setup()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._con = duckdb.connect(":memory:")
        self._con.execute("PRAGMA threads=4;")
        self._create_views()
        self._schema = self._detect_schema()

    def _create_views(self) -> None:
        required = {
            "patients":    self.lake / "patients.parquet",
            "admissions":  self.lake / "admissions.parquet",
            "diagnoses":   self.lake / "diagnoses.parquet",
            "labs":        self.lake / "labs.parquet",
            "medications": self.lake / "medications.parquet",
        }
        optional = {
            "d_icd_diagnoses": self.lake / "d_icd_diagnoses.parquet",
            "d_labitems":      self.lake / "d_labitems.parquet",
            "vitals":          self.lake / "vitals.parquet",
            "notes":           self.lake / "notes.parquet",
            "sparsity":        SPARSITY_FILE,
        }

        missing = [n for n, p in required.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required lakehouse files: {missing}"
            )

        for name, path in required.items():
            self._con.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{path.as_posix()}')"
            )

        for name, path in optional.items():
            if path.exists():
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_parquet('{path.as_posix()}')"
                )
                if name == "sparsity":
                    self._has_sparsity = True
                elif name == "vitals":
                    self._has_vitals = True
                elif name == "notes":
                    self._has_notes = True

    def _detect_schema(self) -> dict:
        def cols(table: str) -> list[str]:
            return [r[0] for r in self._con.execute(f"DESCRIBE {table}").fetchall()]

        def pick(c: list[str], candidates: list[str]) -> str | None:
            low = {x.lower(): x for x in c}
            for cand in candidates:
                if cand.lower() in low:
                    return low[cand.lower()]
            return None

        pc = cols("patients")
        ac = cols("admissions")
        dc = cols("diagnoses")
        lc = cols("labs")
        mc = cols("medications")

        schema: dict = {
            "patients": {
                "subject_id": pick(pc, ["subject_id", "subjectid"]),
                "gender":     pick(pc, ["gender"]),
                "anchor_age": pick(pc, ["anchor_age", "anchorage", "age"]),
            },
            "admissions": {
                "hadm_id":    pick(ac, ["hadm_id", "hadmid"]),
                "subject_id": pick(ac, ["subject_id", "subjectid"]),
                "admittime":  pick(ac, ["admittime"]),
                "dischtime":  pick(ac, ["dischtime"]),
                "diagnosis":  pick(ac, ["diagnosis"]),
            },
            "diagnoses": {
                "hadm_id":     pick(dc, ["hadm_id", "hadmid"]),
                "icd_code":    pick(dc, ["icd_code", "icdcode"]),
                "icd_version": pick(dc, ["icd_version", "icdversion"]),
                "seq_num":     pick(dc, ["seq_num", "seqnum"]),
            },
            "labs": {
                "hadm_id":   pick(lc, ["hadm_id", "hadmid"]),
                "itemid":    pick(lc, ["itemid"]),
                "charttime": pick(lc, ["charttime"]),
                "valuenum":  pick(lc, ["valuenum"]),
                "valueuom":  pick(lc, ["valueuom"]),
                "flag":      pick(lc, ["flag"]),
            },
            "medications": {
                "hadm_id": pick(mc, ["hadm_id", "hadmid"]),
                "drug":    pick(mc, ["drug"]),
                "dose":    pick(mc, ["dose_val_rx", "dose", "dose_val"]),
                "route":   pick(mc, ["route"]),
            },
        }

        # Optional lookup schemas
        views = {r[0].lower() for r in self._con.execute("SHOW TABLES").fetchall()}

        if "d_icd_diagnoses" in views:
            ic = cols("d_icd_diagnoses")
            schema["d_icd_diagnoses"] = {
                "icd_code":    pick(ic, ["icd_code", "icdcode"]),
                "icd_version": pick(ic, ["icd_version", "icdversion"]),
                "long_title":  pick(ic, ["long_title", "longtitle", "title"]),
            }
        else:
            schema["d_icd_diagnoses"] = {
                "icd_code": None, "icd_version": None, "long_title": None
            }

        if "d_labitems" in views:
            li = cols("d_labitems")
            schema["d_labitems"] = {
                "itemid": pick(li, ["itemid"]),
                "label":  pick(li, ["label"]),
            }
        else:
            schema["d_labitems"] = {"itemid": None, "label": None}

        if self._has_vitals:
            vc = cols("vitals")
            schema["vitals"] = {
                "hadm_id":   pick(vc, ["hadm_id", "hadmid"]),
                "itemid":    pick(vc, ["itemid"]),
                "valuenum":  pick(vc, ["valuenum"]),
                "valueuom":  pick(vc, ["valueuom"]),
                "charttime": pick(vc, ["charttime"]),
            }

        return schema

    # ── Core extractors ────────────────────────────────────────────────────────

    def _get_demographics(self, hadm_id: int) -> dict:
        a = self._schema["admissions"]
        p = self._schema["patients"]
        row = self._con.execute(f"""
            SELECT
                a.{a['subject_id']}  AS subject_id,
                p.{p['anchor_age']}  AS age,
                p.{p['gender']}      AS gender
            FROM admissions a
            JOIN patients p ON a.{a['subject_id']} = p.{p['subject_id']}
            WHERE a.{a['hadm_id']} = ?
            LIMIT 1
        """, [hadm_id]).fetchone()

        if row is None:
            return {"subject_id": None, "age": None, "gender": None}

        subject_id, age, gender_raw = row
        g = str(gender_raw).strip().lower() if gender_raw else None
        gender = "male" if g == "m" else "female" if g == "f" else gender_raw
        return {
            "subject_id": int(subject_id) if subject_id else None,
            "age":        int(age)         if age        else None,
            "gender":     gender,
        }

    def _get_diagnoses(self, hadm_id: int) -> list[dict]:
        d  = self._schema["diagnoses"]
        di = self._schema["d_icd_diagnoses"]
        has_lkp = di["icd_code"] and di["long_title"]

        if has_lkp:
            join_on = (
                f"d.{d['icd_code']} = di.{di['icd_code']} "
                f"AND d.{d['icd_version']} = di.{di['icd_version']}"
                if d["icd_version"] and di["icd_version"]
                else f"d.{d['icd_code']} = di.{di['icd_code']}"
            )
            title_expr = f"COALESCE(di.{di['long_title']}, CAST(d.{d['icd_code']} AS VARCHAR))"
            sql = f"""
                SELECT
                    CAST(d.{d['icd_code']} AS VARCHAR) AS icd_code,
                    {title_expr} AS description
                FROM diagnoses d
                LEFT JOIN d_icd_diagnoses di ON {join_on}
                WHERE d.{d['hadm_id']} = ?
                ORDER BY d.{d['seq_num']} ASC NULLS LAST
                LIMIT {MAX_DIAGNOSES}
            """
        else:
            sql = f"""
                SELECT
                    CAST(d.{d['icd_code']} AS VARCHAR) AS icd_code,
                    CAST(d.{d['icd_code']} AS VARCHAR) AS description
                FROM diagnoses d
                WHERE d.{d['hadm_id']} = ?
                ORDER BY d.{d['seq_num']} ASC NULLS LAST
                LIMIT {MAX_DIAGNOSES}
            """

        rows = self._con.execute(sql, [hadm_id]).fetchall()
        return [
            {"icd_code": r[0], "description": str(r[1]).strip()}
            for r in rows if r[0]
        ]

    def _get_labs(self, hadm_id: int) -> list[dict]:
        l  = self._schema["labs"]
        li = self._schema["d_labitems"]
        has_lkp = li["itemid"] and li["label"]

        label_expr  = (f"COALESCE(li.{li['label']}, CAST(l.{l['itemid']} AS VARCHAR))"
                       if has_lkp else f"CAST(l.{l['itemid']} AS VARCHAR)")
        join_clause = (f"LEFT JOIN d_labitems li ON l.{l['itemid']} = li.{li['itemid']}"
                       if has_lkp else "")
        time_expr   = f"l.{l['charttime']}" if l["charttime"] else "NULL"
        flag_expr   = (f"LOWER(COALESCE(CAST(l.{l['flag']} AS VARCHAR), ''))"
                       if l["flag"] else "''")
        uom_expr    = (f"COALESCE(CAST(l.{l['valueuom']} AS VARCHAR), '')"
                       if l["valueuom"] else "''")

        sql = f"""
        WITH ranked AS (
            SELECT
                {label_expr}   AS label,
                l.{l['valuenum']} AS value,
                {uom_expr}     AS unit,
                {flag_expr}    AS flag_text,
                ROW_NUMBER() OVER (
                    PARTITION BY {label_expr}
                    ORDER BY
                        CASE WHEN {flag_expr} = 'abnormal' THEN 0 ELSE 1 END,
                        {time_expr} DESC NULLS LAST
                ) AS rn
            FROM labs l
            {join_clause}
            WHERE l.{l['hadm_id']} = ?
              AND l.{l['valuenum']} IS NOT NULL
        )
        SELECT label, value, unit, flag_text
        FROM ranked
        WHERE rn = 1
          AND label IS NOT NULL
          AND LENGTH(TRIM(CAST(label AS VARCHAR))) > 0
        ORDER BY
            CASE WHEN flag_text = 'abnormal' THEN 0 ELSE 1 END,
            label ASC
        LIMIT {MAX_LABS}
        """

        rows = self._con.execute(sql, [hadm_id]).fetchall()
        return [
            {
                "label": str(r[0]).strip(),
                "value": float(r[1]),
                "unit":  str(r[2]).strip() if r[2] else "",
                "flag":  "abnormal" if str(r[3]).strip() == "abnormal" else "normal",
            }
            for r in rows if r[0] and r[1] is not None
        ]

    def _get_vitals(self, hadm_id: int) -> list[dict]:
        if not self._has_vitals:
            return []

        v = self._schema.get("vitals", {})
        if not v.get("hadm_id") or not v.get("itemid") or not v.get("valuenum"):
            return []

        vital_ids = list(VITAL_ITEMIDS.keys())
        vital_ids_str = ", ".join(str(i) for i in vital_ids)
        time_expr = f"v.{v['charttime']}" if v.get("charttime") else "NULL"
        uom_expr  = (f"COALESCE(CAST(v.{v['valueuom']} AS VARCHAR), '')"
                     if v.get("valueuom") else "''")

        sql = f"""
        WITH ranked AS (
            SELECT
                v.{v['itemid']}    AS itemid,
                v.{v['valuenum']}  AS value,
                {uom_expr}         AS unit,
                ROW_NUMBER() OVER (
                    PARTITION BY v.{v['itemid']}
                    ORDER BY {time_expr} DESC NULLS LAST
                ) AS rn
            FROM vitals v
            WHERE v.{v['hadm_id']} = ?
              AND v.{v['itemid']} IN ({vital_ids_str})
              AND v.{v['valuenum']} IS NOT NULL
        )
        SELECT itemid, value, unit
        FROM ranked
        WHERE rn = 1
        ORDER BY itemid
        LIMIT {MAX_VITALS}
        """

        rows = self._con.execute(sql, [hadm_id]).fetchall()
        return [
            {
                "label": VITAL_ITEMIDS.get(int(r[0]), f"Item {r[0]}"),
                "value": float(r[1]),
                "unit":  str(r[2]).strip() if r[2] else "",
            }
            for r in rows if r[0] and r[1] is not None
        ]

    def _get_medications(self, hadm_id: int) -> list[str]:
        m = self._schema["medications"]
        dose_expr  = (f"CAST(m.{m['dose']} AS VARCHAR)"
                      if m.get("dose") else "NULL")
        route_expr = (f"CAST(m.{m['route']} AS VARCHAR)"
                      if m.get("route") else "NULL")

        sql = f"""
        SELECT DISTINCT
            CAST(m.{m['drug']} AS VARCHAR)  AS drug,
            {dose_expr}                      AS dose,
            {route_expr}                     AS route
        FROM medications m
        WHERE m.{m['hadm_id']} = ?
          AND m.{m['drug']} IS NOT NULL
          AND LENGTH(TRIM(CAST(m.{m['drug']} AS VARCHAR))) > 0
        ORDER BY drug
        LIMIT {MAX_MEDS}
        """

        rows = self._con.execute(sql, [hadm_id]).fetchall()
        meds = []
        for drug, dose, route in rows:
            if not drug:
                continue
            parts = [str(drug).strip()]
            if dose and str(dose).strip() not in ("", "None"):
                parts.append(str(dose).strip())
            if route and str(route).strip() not in ("", "None"):
                parts.append(f"({str(route).strip()})")
            meds.append(" ".join(parts))
        return meds

    def _get_sparsity(self, hadm_id: int) -> dict:
        if not self._has_sparsity:
            return {"sparsity_score": None, "sparsity_bucket": "unknown"}

        row = self._con.execute("""
            SELECT sparsity_score, sparsity_bucket
            FROM sparsity
            WHERE hadm_id = ?
            LIMIT 1
        """, [hadm_id]).fetchone()

        if row is None:
            return {"sparsity_score": None, "sparsity_bucket": "unknown"}

        return {
            "sparsity_score":  float(row[0]) if row[0] is not None else None,
            "sparsity_bucket": str(row[1]) if row[1] else "unknown",
        }

    # ── Text formatter ─────────────────────────────────────────────────────────

    @staticmethod
    def format_snapshot_text(snap: dict) -> str:
        """
        Convert snapshot dict to a formatted string for LLM prompts.
        Example output:
            Patient: 65-year-old male.
            Diagnoses: Type 2 Diabetes (E11.9); CKD Stage 3 (N18.3).
            Labs [abnormal first]: HbA1c 9.1% (abnormal); Creatinine 2.3mg/dL (abnormal).
            Vitals: Heart Rate 88bpm; SpO2 97%.
            Medications: Metformin 1000mg (PO); Lisinopril 10mg (PO).
        """
        lines: list[str] = []

        # Demographics
        age    = snap.get("age")
        gender = snap.get("gender")
        if age and gender:
            lines.append(f"Patient: {age}-year-old {gender}.")
        elif age:
            lines.append(f"Patient: {age}-year-old.")
        elif gender:
            lines.append(f"Patient: {gender}.")

        # Diagnoses
        diags = snap.get("diagnoses", [])
        if diags:
            dx_parts = [
                f"{d['description']} ({d['icd_code']})" for d in diags
            ]
            lines.append("Diagnoses: " + "; ".join(dx_parts) + ".")

        # Labs
        labs = snap.get("labs", [])
        if labs:
            lab_parts = []
            for lab in labs:
                unit  = lab["unit"] or ""
                flag  = " [ABNORMAL]" if lab["flag"] == "abnormal" else ""
                lab_parts.append(f"{lab['label']} {lab['value']}{unit}{flag}")
            lines.append("Labs: " + "; ".join(lab_parts) + ".")

        # Vitals
        vitals = snap.get("vitals", [])
        if vitals:
            vit_parts = [
                f"{v['label']} {v['value']}{v['unit']}" for v in vitals
            ]
            lines.append("Vitals: " + "; ".join(vit_parts) + ".")

        # Medications
        meds = snap.get("medications", [])
        if meds:
            lines.append("Medications: " + "; ".join(meds) + ".")

        # Sparsity (informational, not shown to LLM — stored separately)
        bucket = snap.get("sparsity_bucket", "unknown")
        if bucket and bucket != "unknown":
            lines.append(f"[EHR Sparsity: {bucket}]")

        return "\n".join(lines) if lines else "No structured EHR data available."

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, hadm_id: int) -> dict:
        """
        Main entry point. Returns a complete snapshot dict including
        formatted text string ready for LLM prompt injection.
        """
        demo     = self._get_demographics(hadm_id)
        diags    = self._get_diagnoses(hadm_id)
        labs     = self._get_labs(hadm_id)
        vitals   = self._get_vitals(hadm_id)
        meds     = self._get_medications(hadm_id)
        sparsity = self._get_sparsity(hadm_id)

        snap: dict = {
            "hadm_id":         hadm_id,
            "subject_id":      demo["subject_id"],
            "age":             demo["age"],
            "gender":          demo["gender"],
            "diagnoses":       diags,
            "labs":            labs,
            "vitals":          vitals,
            "medications":     meds,
            "sparsity_score":  sparsity["sparsity_score"],
            "sparsity_bucket": sparsity["sparsity_bucket"],
        }

        snap["snapshot_text"] = self.format_snapshot_text(snap)
        return snap

    def get_batch(self, hadm_ids: list[int]) -> list[dict]:
        """Convenience: get snapshots for a list of admission IDs."""
        return [self.get(h) for h in hadm_ids]

    def close(self) -> None:
        if self._con:
            self._con.close()
            self._con = None


# ── Standalone test ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test get_patient_snapshot for one or more admissions."
    )
    parser.add_argument(
        "--hadm-id", type=int, nargs="+",
        help="One or more hadm_id values to test. "
             "If omitted, picks 5 random ones from ehrqa_finetune.parquet."
    )
    args = parser.parse_args()

    # Resolve test IDs
    if args.hadm_id:
        test_ids = args.hadm_id
    else:
        qa_file = Path("data/qa/ehrqa_finetune.parquet")
        if not qa_file.exists():
            qa_file = Path("data/qa/ehrqa_router_train.parquet")
        df = pd.read_parquet(qa_file, columns=["hadm_id"])
        test_ids = df["hadm_id"].drop_duplicates().sample(
            min(5, len(df)), random_state=42
        ).tolist()

    print(f"[INFO] Testing PatientSnapshot for hadm_ids: {test_ids}")
    print()

    api = PatientSnapshot()

    pass_count = 0
    for hadm_id in test_ids:
        t0   = time.time()
        snap = api.get(hadm_id)
        elapsed = (time.time() - t0) * 1000

        print("─" * 60)
        print(f"hadm_id     : {snap['hadm_id']}")
        print(f"subject_id  : {snap['subject_id']}")
        print(f"age/gender  : {snap['age']} / {snap['gender']}")
        print(f"diagnoses   : {len(snap['diagnoses'])} entries")
        print(f"labs        : {len(snap['labs'])} entries")
        print(f"vitals      : {len(snap['vitals'])} entries")
        print(f"medications : {len(snap['medications'])} entries")
        print(f"sparsity    : score={snap['sparsity_score']} bucket={snap['sparsity_bucket']}")
        print(f"latency     : {elapsed:.1f}ms")
        print()
        print("── Formatted Snapshot Text ──")
        print(snap["snapshot_text"])
        print()

        # Gate check
        has_data = (
            snap["diagnoses"] or snap["labs"] or snap["medications"]
        )
        if has_data:
            pass_count += 1
        else:
            print(f"  ⚠️  WARNING: No structured data found for hadm_id {hadm_id}")

    api.close()

    print("═" * 60)
    print(f"Gate G1 check: {pass_count}/{len(test_ids)} snapshots have structured data.")
    if pass_count == len(test_ids):
        print("✅ PASS — get_patient_snapshot() working correctly.")
    else:
        print("⚠️  Some snapshots returned empty. Check lakehouse completeness.")


if __name__ == "__main__":
    main()