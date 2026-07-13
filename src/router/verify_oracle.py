import pandas as pd

TRAIN_FILE = "data/router/router_train_oracle.parquet"
VAL_FILE = "data/router/router_val_oracle.parquet"


def verify_basic(path):
    print("=" * 80)
    print(path)
    print("=" * 80)

    df = pd.read_parquet(path)

    print("\nShape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nBest Mode Distribution:")
    print(df["best_mode"].value_counts())

    print("\nPercentage:")
    print((df["best_mode"].value_counts(normalize=True) * 100).round(2))

    return df


def verify_oracle_consistency(df):
    print("\n" + "=" * 80)
    print("CHECK 2 : ORACLE CONSISTENCY")
    print("=" * 80)

    correct = 0

    sample = df.sample(min(20, len(df)), random_state=42)

    for _, row in sample.iterrows():

        scores = {
            "T": row["composite_t"],
            "T+E": row["composite_te"],
            "T+E+K": row["composite_tek"],
        }

        predicted = max(scores, key=scores.get)

        if predicted == row["best_mode"]:
            correct += 1

        print("-" * 60)
        print(f"Stored Oracle : {row['best_mode']}")
        print(f"Calculated    : {predicted}")
        print(scores)

    print("\nConsistency:")
    print(f"{correct}/{len(sample)} ({correct/len(sample)*100:.1f}%)")


def verify_hallucinations(df):
    print("\n" + "=" * 80)
    print("CHECK 3 : HALLUCINATION STATISTICS")
    print("=" * 80)

    cols = [
        "halluc_t",
        "halluc_te",
        "halluc_tek",
    ]

    print(df[cols].sum())


def verify_latency(df):
    print("\n" + "=" * 80)
    print("CHECK 4 : LATENCY")
    print("=" * 80)

    print(
        df[
            [
                "latency_t",
                "latency_te",
                "latency_tek",
            ]
        ].describe()
    )


def manual_inspection(df):
    print("\n" + "=" * 80)
    print("CHECK 5 : RANDOM MANUAL INSPECTION")
    print("=" * 80)

    sample = df.sample(5, random_state=7)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):

        print("\n" + "=" * 80)
        print(f"SAMPLE {i}")
        print("=" * 80)

        print("\nQuestion:")
        print(row["question"])

        print("\nGround Truth:")
        print(row["reference_answer"])

        print("\nAnswer T:")
        print(row["answer_t"])

        print("\nAnswer T+E:")
        print(row["answer_te"])

        print("\nAnswer T+E+K:")
        print(row["answer_tek"])

        print("\nOracle Label:")
        print(row["best_mode"])

        print("\nComposite Scores:")

        print(
            {
                "T": row["composite_t"],
                "T+E": row["composite_te"],
                "T+E+K": row["composite_tek"],
            }
        )


def main():

    train_df = verify_basic(TRAIN_FILE)
    val_df = verify_basic(VAL_FILE)

    verify_oracle_consistency(train_df)

    verify_hallucinations(train_df)

    verify_latency(train_df)

    manual_inspection(train_df)

    print("\n" + "=" * 80)
    print("FINAL STATUS")
    print("=" * 80)
    print("Oracle verification completed successfully.")


if __name__ == "__main__":
    main()