from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "intent_classifier.pkl"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_set.csv"


def main():

    print("=" * 70)
    print("INTENT CONFUSION ANALYSIS")
    print("=" * 70)

    # Load model
    print("\nLoading classifier...")
    model = joblib.load(MODEL_PATH)

    # Load golden set
    print("Loading golden set...")
    df = pd.read_csv(GOLDEN_PATH)

    df = df.dropna(subset=["customer_text", "intent"])

    X = df["customer_text"].astype(str)
    y_true = df["intent"].astype(str)

    # Predictions
    print("Running predictions...")
    y_pred = model.predict(X)

    # Get all intent labels
    labels = sorted(set(y_true) | set(y_pred))

    # Confusion matrix
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print("\nRows = Expected")
    print("Columns = Predicted\n")

    print(confusion_df.to_string())

    # Show only incorrect prediction pairs
    print("\n" + "-" * 70)
    print("TOP CONFUSION PAIRS")
    print("-" * 70)

    pairs = {}

    for actual, predicted in zip(y_true, y_pred):

        if actual != predicted:

            pair = (actual, predicted)

            if pair not in pairs:
                pairs[pair] = 0

            pairs[pair] += 1

    sorted_pairs = sorted(
        pairs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for (actual, predicted), count in sorted_pairs:

        print(
            f"{actual:20} -> "
            f"{predicted:20} : {count}"
        )

    # Save results
    output_path = PROJECT_ROOT / "evaluation" / "confusion_matrix.csv"
    confusion_df.to_csv(output_path)

    print("\n" + "-" * 70)
    print(f"Confusion matrix saved to:")
    print(output_path)

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()