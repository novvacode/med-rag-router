"""
ehr_snapshot.py — Assemble a per-patient EHR snapshot as natural language
context for the T+E (text + structured EHR) retrieval mode.

Pulls diagnoses, most recent labs, and medications for a given hadm_id
from the lakehouse and formats them into a readable clinical summary,
matching the format in project spec Section 5.4.
"""

import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from lakehouse.query import get_connection


def get_patient_demographics(con, hadm_id: int) -> dict:
    q = """
        SELECT p.gender, p.anchor_age AS age
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.hadm_id = $hadm_id
    """
    df = con.execute(q, {"hadm_id": hadm_id}).fetchdf()
    if df.empty:
        return {"age": "unknown", "gender": "unknown"}
    row = df.iloc[0]
    gender_map = {"M": "male", "F": "female"}
    return {"age": int(row["age"]), "gender": gender_map.get(row["gender"], row["gender"])}


def get_diagnoses(con, hadm_id: int, max_dx: int = 5) -> list:
    q = """
        SELECT DISTINCT d.icd_code, di.long_title
        FROM diagnoses d
        LEFT JOIN d_icd_diagnoses di ON d.icd_code = di.icd_code
        WHERE d.hadm_id = $hadm_id
        LIMIT $max_dx
    """
    df = con.execute(q, {"hadm_id": hadm_id, "max_dx": max_dx}).fetchdf()
    return df["long_title"].dropna().tolist() if not df.empty else []


def get_recent_labs(con, hadm_id: int, max_labs: int = 8) -> list:
    """
    Get the most recent abnormal-flagged labs first, then fill with normal ones.
    Filters out malformed rows: labels that are too short to be real lab names,
    or unit strings that are literally the text 'nan' (bad joins/type coercion).
    """
    q = """
        SELECT li.label, l.valuenum, l.valueuom, l.flag, l.charttime
        FROM labs l
        LEFT JOIN d_labitems li ON l.itemid = li.itemid
        WHERE l.hadm_id = $hadm_id AND l.valuenum IS NOT NULL AND li.label IS NOT NULL
        ORDER BY (l.flag = 'abnormal') DESC, l.charttime DESC
        LIMIT 40
    """
    df = con.execute(q, {"hadm_id": hadm_id}).fetchdf()
    if df.empty:
        return []

    df["valueuom"] = df["valueuom"].astype(str)
    df = df[df["label"].str.len() > 2]
    df = df[~df["valueuom"].str.lower().isin(["nan", "none"])]
    df = df[df["label"].str.lower() != "nan"]

    df = df.drop_duplicates(subset=["label"]).head(max_labs)

    labs = []
    for _, row in df.iterrows():
        flag_str = " (abnormal)" if row["flag"] == "abnormal" else ""
        unit = row["valueuom"] if row["valueuom"] not in ("nan", "None") else ""
        labs.append(f"{row['label']} {row['valuenum']}{unit}{flag_str}")
    return labs


def get_medications(con, hadm_id: int, max_meds: int = 5) -> list:
    q = """
        SELECT DISTINCT drug
        FROM medications
        WHERE hadm_id = $hadm_id AND drug IS NOT NULL
        LIMIT $max_meds
    """
    df = con.execute(q, {"hadm_id": hadm_id, "max_meds": max_meds}).fetchdf()
    return df["drug"].dropna().tolist() if not df.empty else []


def format_ehr_snapshot(hadm_id: int) -> str:
    """Main entry point: build the natural-language EHR snapshot for a hadm_id."""
    con = get_connection()

    demo = get_patient_demographics(con, hadm_id)
    diagnoses = get_diagnoses(con, hadm_id)
    labs = get_recent_labs(con, hadm_id)
    meds = get_medications(con, hadm_id)

    parts = [f"Patient: {demo['age']}-year-old {demo['gender']}."]

    if diagnoses:
        parts.append(f"Diagnoses: {', '.join(diagnoses)}.")
    else:
        parts.append("Diagnoses: none recorded.")

    if labs:
        parts.append(f"Labs: {', '.join(labs)}.")
    else:
        parts.append("Labs: none recorded.")

    if meds:
        parts.append(f"Medications: {', '.join(meds)}.")
    else:
        parts.append("Medications: none recorded.")

    return " ".join(parts)


if __name__ == "__main__":
    con = get_connection()
    sample_hadm_ids = con.execute(
        "SELECT hadm_id FROM admissions ORDER BY RANDOM() LIMIT 3"
    ).fetchdf()["hadm_id"].tolist()

    print(f"Testing with sample hadm_ids: {sample_hadm_ids}\n")
    for hadm_id in sample_hadm_ids:
        print(f"--- hadm_id: {hadm_id} ---")
        snapshot = format_ehr_snapshot(int(hadm_id))
        print(snapshot)
        print()