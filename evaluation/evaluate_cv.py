from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_set.csv"


def main():

    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    # Load golden set
    print("\nLoading golden evaluation set...")
    df = pd.read_csv(GOLDEN_PATH)

    df = df.dropna(subset=["customer_text", "intent"])

    X = df["customer_text"].astype(str)
    y = df["intent"].astype(str)

    print(f"Total examples: {len(df)}")
    print(f"Number of intents: {y.nunique()}")

    # TF-IDF + Logistic Regression
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("\nRunning 5-fold cross-validation...")

    predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv
    )

    # Metrics
    accuracy = accuracy_score(y, predictions)

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    # Detailed report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    # Incorrect predictions
    results = df.copy()

    results["predicted_intent"] = predictions

    errors = results[
        results["intent"] != results["predicted_intent"]
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
    print("Cross-validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()