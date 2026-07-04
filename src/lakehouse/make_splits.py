"""
make_splits.py — Create leakage-safe patient ID splits.

Splits (per project spec):
  - finetune_train   : patients used for QLoRA fine-tuning (synthetic EHR-QA)
  - router_train      : ~200 questions worth of patients, for router pseudo-labels
  - router_val         : ~100 questions worth of patients, for router hyperparameter tuning
  - held_out_eval      : 200-300 questions worth of patients, FINAL evaluation only

All splits are disjoint at the PATIENT level (subject_id), not just admission level,
to strictly prevent any data leakage.
"""

import sys
import json
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lakehouse.query import get_connection, get_patient_ids

RANDOM_SEED = 42
OUTPUT_PATH = Path("splits/patient_splits.json")

SPLIT_RATIOS = {
    "finetune_train": 0.70,
    "router_train":   0.10,
    "router_val":     0.05,
    "held_out_eval":  0.15,
}

assert abs(sum(SPLIT_RATIOS.values()) - 1.0) < 1e-9


def main():
    con = get_connection()
    all_ids = get_patient_ids(con)
    print(f"Total unique patients in lakehouse: {len(all_ids)}")

    random.seed(RANDOM_SEED)
    shuffled = all_ids.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    splits = {}
    start = 0
    for name, ratio in SPLIT_RATIOS.items():
        count = int(n * ratio)
        splits[name] = shuffled[start:start + count]
        start += count
    if start < n:
        splits["finetune_train"].extend(shuffled[start:])

    seen = set()
    for name, ids in splits.items():
        overlap = seen.intersection(ids)
        assert not overlap, f"LEAKAGE DETECTED in {name}: {overlap}"
        seen.update(ids)

    total_assigned = sum(len(v) for v in splits.values())
    assert total_assigned == n, "Not all patients were assigned to a split!"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    print("\nSplit sizes (patient counts):")
    for name, ids in splits.items():
        print(f"  {name:16s}: {len(ids):6d} patients")

    print(f"\nNo overlaps detected. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()