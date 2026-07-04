"""
ingest.py — Convert MIMIC-IV raw CSV.gz files into partitioned Parquet
for the healthcare lakehouse. Uses DuckDB for fast, memory-safe conversion
even for large files like labevents.csv.gz (2.5GB+) and discharge.csv.gz (1.1GB).
"""

import duckdb
from pathlib import Path

RAW_DIR = Path("data/raw")
LAKEHOUSE_DIR = Path("data/lakehouse")
LAKEHOUSE_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

def csv_to_parquet(csv_path: Path, parquet_path: Path, columns: str = "*"):
    """Stream a gzipped CSV into a Parquet file using DuckDB (no full RAM load)."""
    print(f"Converting {csv_path.name} -> {parquet_path.name} ...")
    query = f"""
        COPY (
            SELECT {columns} FROM read_csv_auto('{csv_path.as_posix()}', 
                                                   sample_size=-1, 
                                                   ignore_errors=True)
        ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  -> Done. {parquet_path.name}: {size_mb:.1f} MB")


def main():
    hosp = RAW_DIR / "hosp"
    note = RAW_DIR / "note"

    # 1. Patients
    csv_to_parquet(hosp / "patients.csv.gz", LAKEHOUSE_DIR / "patients.parquet")

    # 2. Admissions
    csv_to_parquet(hosp / "admissions.csv.gz", LAKEHOUSE_DIR / "admissions.parquet")

    # 3. Diagnoses (ICD codes per admission)
    csv_to_parquet(hosp / "diagnoses_icd.csv.gz", LAKEHOUSE_DIR / "diagnoses.parquet")

    # 4. ICD dictionary (code -> description)
    csv_to_parquet(hosp / "d_icd_diagnoses.csv.gz", LAKEHOUSE_DIR / "d_icd_diagnoses.parquet")

    # 5. Labs (BIG FILE — streamed, not loaded into RAM)
    csv_to_parquet(hosp / "labevents.csv.gz", LAKEHOUSE_DIR / "labs.parquet",
                   columns="subject_id, hadm_id, itemid, charttime, valuenum, valueuom, flag")

    # 6. Lab item dictionary
    csv_to_parquet(hosp / "d_labitems.csv.gz", LAKEHOUSE_DIR / "d_labitems.parquet")

    # 7. Medications
    csv_to_parquet(hosp / "prescriptions.csv.gz", LAKEHOUSE_DIR / "medications.parquet",
                   columns="subject_id, hadm_id, drug, starttime, stoptime, dose_val_rx, dose_unit_rx")

    # 8. Clinical notes (discharge summaries)
    csv_to_parquet(note / "discharge.csv.gz", LAKEHOUSE_DIR / "notes.parquet")

    print("\nAll files converted successfully.")
    print("\nLakehouse summary:")
    for f in sorted(LAKEHOUSE_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()