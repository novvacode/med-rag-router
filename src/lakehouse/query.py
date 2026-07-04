"""
query.py — DuckDB query helpers over the Parquet lakehouse.
All functions return pandas DataFrames. Connection is read-only and
lightweight — DuckDB queries Parquet files directly without loading
everything into memory.
"""

import duckdb
from pathlib import Path

LAKEHOUSE_DIR = Path("data/lakehouse")

def get_connection():
    """Fresh in-memory DuckDB connection. Views map to Parquet files on disk."""
    con = duckdb.connect()
    for name in ["patients", "admissions", "diagnoses", "d_icd_diagnoses",
                 "labs", "d_labitems", "medications", "notes"]:
        path = (LAKEHOUSE_DIR / f"{name}.parquet").as_posix()
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con


def get_patient_ids(con, limit=None) -> list:
    """All distinct subject_ids in the dataset."""
    q = "SELECT DISTINCT subject_id FROM patients"
    if limit:
        q += f" LIMIT {limit}"
    return con.execute(q).fetchdf()["subject_id"].tolist()


def get_patient_demographics(con, subject_id: int):
    return con.execute(f"""
        SELECT subject_id, gender, anchor_age, anchor_year
        FROM patients WHERE subject_id = {subject_id}
    """).fetchdf()


def get_patient_admissions(con, subject_id: int):
    return con.execute(f"""
        SELECT hadm_id, admittime, dischtime, admission_type, discharge_location
        FROM admissions WHERE subject_id = {subject_id}
        ORDER BY admittime
    """).fetchdf()


def get_patient_diagnoses(con, subject_id: int, hadm_id: int = None):
    where = f"subject_id = {subject_id}"
    if hadm_id:
        where += f" AND hadm_id = {hadm_id}"
    return con.execute(f"""
        SELECT d.subject_id, d.hadm_id, d.icd_code, dx.long_title
        FROM diagnoses d
        LEFT JOIN d_icd_diagnoses dx ON d.icd_code = dx.icd_code AND d.icd_version = dx.icd_version
        WHERE {where}
        ORDER BY d.seq_num
    """).fetchdf()


def get_patient_labs(con, subject_id: int, hadm_id: int = None, last_n_days: int = None):
    where = f"l.subject_id = {subject_id}"
    if hadm_id:
        where += f" AND l.hadm_id = {hadm_id}"
    return con.execute(f"""
        SELECT l.subject_id, l.hadm_id, l.charttime, li.label, l.valuenum, l.valueuom, l.flag
        FROM labs l
        LEFT JOIN d_labitems li ON l.itemid = li.itemid
        WHERE {where}
        ORDER BY l.charttime DESC
    """).fetchdf()


def get_patient_medications(con, subject_id: int, hadm_id: int = None):
    where = f"subject_id = {subject_id}"
    if hadm_id:
        where += f" AND hadm_id = {hadm_id}"
    return con.execute(f"""
        SELECT subject_id, hadm_id, drug, starttime, stoptime, dose_val_rx, dose_unit_rx
        FROM medications WHERE {where}
        ORDER BY starttime
    """).fetchdf()


def get_patient_notes(con, subject_id: int, hadm_id: int = None):
    where = f"subject_id = {subject_id}"
    if hadm_id:
        where += f" AND hadm_id = {hadm_id}"
    return con.execute(f"""
        SELECT subject_id, hadm_id, charttime, text
        FROM notes WHERE {where}
        ORDER BY charttime DESC
    """).fetchdf()


if __name__ == "__main__":
    con = get_connection()
    pids = get_patient_ids(con, limit=5)
    print("Sample patient IDs:", pids)

    test_id = pids[0]
    print(f"\n--- Demographics for {test_id} ---")
    print(get_patient_demographics(con, test_id))

    print(f"\n--- Admissions for {test_id} ---")
    print(get_patient_admissions(con, test_id))

    print(f"\n--- Diagnoses for {test_id} ---")
    print(get_patient_diagnoses(con, test_id).head())

    print(f"\n--- Labs for {test_id} ---")
    print(get_patient_labs(con, test_id).head())

    print(f"\n--- Medications for {test_id} ---")
    print(get_patient_medications(con, test_id).head())

    print(f"\n--- Notes for {test_id} ---")
    notes_df = get_patient_notes(con, test_id)
    print(f"Found {len(notes_df)} notes")
    if len(notes_df) > 0:
        print(notes_df["text"].iloc[0][:300], "...")