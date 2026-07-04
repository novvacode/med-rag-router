"""
snapshot.py — Assemble per-patient/encounter EHR snapshots as natural language context.

This is the core function used by retrieval modes T+E and T+E+K to inject
structured patient data into the LLM prompt.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lakehouse.query import (
    get_connection,
    get_patient_demographics,
    get_patient_diagnoses,
    get_patient_labs,
    get_patient_medications,
)


def _format_labs(labs_df: pd.DataFrame, max_labs: int = 8) -> str:
    """Format most recent abnormal labs first, then normal, capped at max_labs."""
    if labs_df.empty:
        return "No lab results available."

    df = labs_df.dropna(subset=["label"]).copy()
    df["is_abnormal"] = (df["flag"] == "abnormal").astype(int)
    df = df.sort_values(["is_abnormal", "charttime"], ascending=[False, False])
    df = df.drop_duplicates(subset=["label"]).head(max_labs)

    parts = []
    for _, row in df.iterrows():
        val = row["valuenum"]
        unit = row["valueuom"] if pd.notna(row["valueuom"]) else ""
        flag = " (high/abnormal)" if row["flag"] == "abnormal" else ""
        if pd.notna(val):
            parts.append(f"{row['label']} {val}{unit}{flag}")
    return ", ".join(parts) if parts else "No lab results available."


def _format_diagnoses(diag_df: pd.DataFrame, max_dx: int = 6) -> str:
    if diag_df.empty:
        return "No diagnoses recorded."
    titles = diag_df["long_title"].dropna().unique().tolist()[:max_dx]
    return ", ".join(titles) if titles else "No diagnoses recorded."


def _format_medications(meds_df: pd.DataFrame, max_meds: int = 6) -> str:
    if meds_df.empty:
        return "No medications recorded."
    rows = meds_df.drop_duplicates(subset=["drug"]).head(max_meds)
    parts = []
    for _, row in rows.iterrows():
        dose = f" {row['dose_val_rx']}{row['dose_unit_rx']}" if pd.notna(row.get("dose_val_rx")) else ""
        parts.append(f"{row['drug']}{dose}")
    return ", ".join(parts) if parts else "No medications recorded."


def get_patient_snapshot(con, subject_id: int, hadm_id: int = None) -> str:
    """
    Build a natural-language EHR snapshot string for one patient/encounter.
    Matches the format specified in the project doc (Section 5.4).
    """
    demo = get_patient_demographics(con, subject_id)
    if demo.empty:
        return f"No demographic data found for patient {subject_id}."

    age = int(demo["anchor_age"].iloc[0])
    gender = "male" if demo["gender"].iloc[0] == "M" else "female"

    diagnoses = get_patient_diagnoses(con, subject_id, hadm_id)
    labs = get_patient_labs(con, subject_id, hadm_id)
    meds = get_patient_medications(con, subject_id, hadm_id)

    dx_str = _format_diagnoses(diagnoses)
    labs_str = _format_labs(labs)
    meds_str = _format_medications(meds)

    snapshot = (
        f"Patient: {age}-year-old {gender}. "
        f"Diagnoses: {dx_str}. "
        f"Labs: {labs_str}. "
        f"Medications: {meds_str}."
    )
    return snapshot


if __name__ == "__main__":
    con = get_connection()

    test_cases = [
        (10000032, 22595853),
        (10000032, None),  # all encounters combined
    ]

    for subject_id, hadm_id in test_cases:
        print(f"\n=== Snapshot: subject_id={subject_id}, hadm_id={hadm_id} ===")
        snapshot = get_patient_snapshot(con, subject_id, hadm_id)
        print(snapshot)