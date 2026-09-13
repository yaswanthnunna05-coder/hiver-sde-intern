import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_PATH = "data/golden_set.csv"

df = pd.read_csv(GOLDEN_PATH)

y_true = df["intent"]

# --------------------------------------------------
# Baseline 1: Majority-class classifier
# --------------------------------------------------

majority_class = y_true.value_counts().idxmax()

y_pred_majority = [majority_class] * len(y_true)

majority_accuracy = accuracy_score(y_true, y_pred_majority)
majority_macro_f1 = f1_score(
    y_true,
    y_pred_majority,
    average="macro",
    zero_division=0
)

print("=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)

print(f"Golden examples: {len(df)}")
print(f"Majority class: {majority_class}")
print(f"Majority baseline accuracy: {majority_accuracy:.2%}")
print(f"Majority baseline macro F1: {majority_macro_f1:.4f}")

print("\nGolden-set class distribution:")
print(y_true.value_counts())

# --------------------------------------------------
# Our V3 classifier results
# --------------------------------------------------

v3_accuracy = 0.60
v3_macro_f1 = 0.5307

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"Majority baseline accuracy: {majority_accuracy:.2%}")
print(f"V3 classifier accuracy:     {v3_accuracy:.2%}")

print(
    f"\nAccuracy improvement: "
    f"{(v3_accuracy - majority_accuracy):.2%}"
)

print(f"\nMajority baseline macro F1: {majority_macro_f1:.4f}")
print(f"V3 classifier macro F1:     {v3_macro_f1:.4f}")