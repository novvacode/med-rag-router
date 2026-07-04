"""
sparsity.py — Compute EHR sparsity metrics per patient-encounter (H2 operationalization).

Metrics:
  n_labs  = number of distinct lab tests recorded for the encounter
  n_diag  = number of distinct ICD diagnosis codes for the encounter
  d_note  = days since last clinical note before question/discharge time

Composite sparsity score buckets encounters into High / Medium / Low.
"""

import duckdb
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lakehouse.query import get_connection

# Thresholds (from project spec, Section 8)
TAU_LABS = 5
TAU_DIAG = 3
TAU_DAYS = 7

ALPHA_1, ALPHA_2, ALPHA_3 = 1.0, 1.0, 1.0  # equal weights, can be tuned empirically


def compute_sparsity_table(con) -> pd.DataFrame:
    """
    Build one row per (subject_id, hadm_id) encounter with sparsity metrics.
    """
    labs_q = """
        SELECT subject_id, hadm_id, COUNT(DISTINCT itemid) AS n_labs
        FROM labs
        WHERE hadm_id IS NOT NULL
        GROUP BY subject_id, hadm_id
    """

    diag_q = """
        SELECT subject_id, hadm_id, COUNT(DISTINCT icd_code) AS n_diag
        FROM diagnoses
        GROUP BY subject_id, hadm_id
    """

    note_q = """
        SELECT n.subject_id, n.hadm_id, MAX(n.charttime) AS last_note_time
        FROM notes n
        WHERE n.hadm_id IS NOT NULL
        GROUP BY n.subject_id, n.hadm_id
    """

    admissions_q = "SELECT subject_id, hadm_id, dischtime FROM admissions"

    labs_df = con.execute(labs_q).fetchdf()
    diag_df = con.execute(diag_q).fetchdf()
    note_df = con.execute(note_q).fetchdf()
    adm_df = con.execute(admissions_q).fetchdf()

    df = adm_df.merge(labs_df, on=["subject_id", "hadm_id"], how="left")
    df = df.merge(diag_df, on=["subject_id", "hadm_id"], how="left")
    df = df.merge(note_df, on=["subject_id", "hadm_id"], how="left")

    df["n_labs"] = df["n_labs"].fillna(0).astype(int)
    df["n_diag"] = df["n_diag"].fillna(0).astype(int)

    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["last_note_time"] = pd.to_datetime(df["last_note_time"])
    df["d_note"] = (df["dischtime"] - df["last_note_time"]).dt.days
    df["d_note"] = df["d_note"].fillna(999).astype(int)
    df.loc[df["d_note"] < 0, "d_note"] = 0

    df["sparsity_score"] = (
        ALPHA_1 * (df["n_labs"] < TAU_LABS).astype(int)
        + ALPHA_2 * (df["n_diag"] < TAU_DIAG).astype(int)
        + ALPHA_3 * (df["d_note"] > TAU_DAYS).astype(int)
    )

    def bucket(score):
        if score >= 2:
            return "High"
        elif score == 1:
            return "Medium"
        else:
            return "Low"

    df["sparsity_bucket"] = df["sparsity_score"].apply(bucket)

    return df[["subject_id", "hadm_id", "n_labs", "n_diag", "d_note",
               "sparsity_score", "sparsity_bucket"]]


def main():
    con = get_connection()
    df = compute_sparsity_table(con)

    out_path = Path("data/lakehouse/sparsity.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Computed sparsity for {len(df)} encounters.")
    print("\nBucket distribution:")
    print(df["sparsity_bucket"].value_counts())
    print(f"\nSaved to {out_path}")
    print("\nSample rows:")
    print(df.head(10))


if __name__ == "__main__":
    main()