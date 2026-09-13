import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Allow importing files from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

# Paths
MODEL_PATH = PROJECT_ROOT / "models" / "intent_classifier.pkl"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_set.csv"


def main():

    print("=" * 70)
    print("INTENT CLASSIFIER EVALUATION")
    print("=" * 70)

    # Load model
    print("\nLoading trained classifier...")
    model = joblib.load(MODEL_PATH)

    # Load golden set
    print("Loading golden evaluation set...")
    df = pd.read_csv(GOLDEN_PATH)

    print(f"Golden examples: {len(df)}")

    # Check required columns
    required_columns = ["customer_text", "intent"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # Remove rows with missing values
    df = df.dropna(subset=["customer_text", "intent"])

    X = df["customer_text"].astype(str)
    y_true = df["intent"].astype(str)

    # Predict
    print("\nRunning predictions...")
    y_pred = model.predict(X)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1 : {macro_f1:.4f}")

    # Detailed report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # Show incorrect predictions
    df_results = df.copy()
    df_results["predicted_intent"] = y_pred

    errors = df_results[
        df_results["intent"] != df_results["predicted_intent"]
    ]

    print("-" * 70)
    print(f"INCORRECT PREDICTIONS: {len(errors)}")
    print("-" * 70)

    for _, row in errors.head(10).iterrows():

        print("\nCustomer:")
        print(row["customer_text"])

        print(f"Expected : {row['intent']}")
        print(f"Predicted: {row['predicted_intent']}")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()