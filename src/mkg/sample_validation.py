"""
sample_validation.py — Sample 50 random edges from the MKG for manual
clinical validation (Section 7.3 of project spec).
"""

import pandas as pd
from pathlib import Path

ONTOLOGY_PATH = Path("mkg/edges/ontology_edges.csv")
COOCCURRENCE_PATH = Path("mkg/edges/cooccurrence_edges.csv")
OUTPUT_PATH = Path("mkg/validation/edge_validation_50.csv")

def main():
    ontology_df = pd.read_csv(ONTOLOGY_PATH)
    cooc_df = pd.read_csv(COOCCURRENCE_PATH)

    ontology_df["edge_source"] = "ontology"

    cooc_renamed = cooc_df.rename(columns={"lab_test": "target"})
    cooc_renamed["edge_type"] = "CO_OCCURS_WITH_LAB"
    cooc_renamed["target_type"] = "LabTest"
    cooc_renamed["notes"] = cooc_renamed["frequency"].apply(
        lambda f: f"co-occurs in {f:.1%} of admissions"
    )
    cooc_renamed["edge_source"] = "ehr_cooccurrence"

    combined = pd.concat([
        ontology_df[["disease", "edge_type", "target", "target_type", "notes", "edge_source"]],
        cooc_renamed[["disease", "edge_type", "target", "target_type", "notes", "edge_source"]]
    ], ignore_index=True)

    print(f"Total edges available: {len(combined)}")

    sample_50 = combined.sample(n=50, random_state=42).reset_index(drop=True)
    sample_50["edge_id"] = range(1, 51)
    sample_50["validated_correct"] = ""   # Fill in: Yes / No / Partial
    sample_50["reviewer_notes"] = ""      # Optional clinical reference/comment

    sample_50 = sample_50[["edge_id", "disease", "edge_type", "target", "target_type",
                            "notes", "edge_source", "validated_correct", "reviewer_notes"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_50.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved 50-edge validation sample to {OUTPUT_PATH}")
    print("\nSample source breakdown:")
    print(sample_50["edge_source"].value_counts())
    print("\nFirst 10 rows:")
    print(sample_50.head(10).to_string())


if __name__ == "__main__":
    main()