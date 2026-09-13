import pandas as pd

PATH = "evaluation/reply_evaluation_results.csv"

df = pd.read_csv(PATH)

errors = df[
    df["expected_intent"] != df["predicted_intent"]
].copy()

print("=" * 70)
print("FAILURE ANALYSIS")
print("=" * 70)

print(f"Total examples: {len(df)}")
print(f"Incorrect predictions: {len(errors)}")
print(f"Error rate: {len(errors) / len(df):.2%}")

print("\nTop confusion pairs:")
print("-" * 70)

confusions = (
    errors.groupby(
        ["expected_intent", "predicted_intent"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(confusions.to_string(index=False))


print("\n" + "=" * 70)
print("REPRESENTATIVE EXAMPLES")
print("=" * 70)

# Show up to 3 examples for each major confusion pair
for _, row in confusions.head(5).iterrows():

    expected = row["expected_intent"]
    predicted = row["predicted_intent"]

    subset = errors[
        (errors["expected_intent"] == expected) &
        (errors["predicted_intent"] == predicted)
    ].head(3)

    print(
        f"\n{expected} -> {predicted} "
        f"({len(subset)} shown)"
    )
    print("-" * 70)

    for _, example in subset.iterrows():

        print("Customer:")
        print(example["customer_text"])

        print("\nPredicted:")
        print(example["predicted_intent"])

        print("Expected:")
        print(example["expected_intent"])

        print()