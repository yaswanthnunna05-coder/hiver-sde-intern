import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score

# Allow importing from src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from train_classifier import assign_intent


GOLDEN_PATH = "data/golden_set.csv"
CONVERSATIONS_PATH = "data/apple_conversations.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

golden = pd.read_csv(GOLDEN_PATH)
history = pd.read_csv(CONVERSATIONS_PATH)

golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

history["customer_tweet_id"] = (
    history["customer_tweet_id"].astype(str)
)

# Remove golden examples to prevent leakage
history = history[
    ~history["customer_tweet_id"].isin(golden_ids)
].copy()

history["customer_text"] = (
    history["customer_text"]
    .fillna("")
    .astype(str)
)


print("=" * 60)
print("TF-IDF RETRIEVAL BASELINE")
print("=" * 60)

print(f"Golden examples: {len(golden)}")
print(f"Historical examples: {len(history)}")


# --------------------------------------------------
# Assign V3 intent labels to historical messages
# --------------------------------------------------

print("\nApplying V3 intent rules to historical messages...")

history["historical_intent"] = (
    history["customer_text"]
    .apply(assign_intent)
)

print("Historical intent labels ready.")


# --------------------------------------------------
# Build TF-IDF index
# --------------------------------------------------

print("\nBuilding TF-IDF search index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

history_vectors = vectorizer.fit_transform(
    history["customer_text"]
)

print("Search index ready.")


# --------------------------------------------------
# Retrieve nearest historical example
# --------------------------------------------------

predictions = []
similarities = []
matched_customer_text = []
matched_intent = []


for i, row in golden.iterrows():

    query = str(row["customer_text"])

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        history_vectors
    )[0]

    best_index = scores.argmax()

    best_row = history.iloc[best_index]

    predictions.append(
        best_row["historical_intent"]
    )

    similarities.append(
        float(scores[best_index])
    )

    matched_customer_text.append(
        best_row["customer_text"]
    )

    matched_intent.append(
        best_row["historical_intent"]
    )


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

golden["retrieval_prediction"] = predictions
golden["retrieval_similarity"] = similarities
golden["matched_customer_text"] = matched_customer_text
golden["matched_intent"] = matched_intent


y_true = golden["intent"]
y_pred = golden["retrieval_prediction"]


accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

average_similarity = sum(similarities) / len(similarities)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.2%}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Average similarity: {average_similarity:.3f}")


print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

print("Majority baseline accuracy: 48.67%")
print(f"TF-IDF retrieval accuracy:  {accuracy:.2%}")
print("V3 classifier accuracy:     61.33%")

print(
    f"\nRetrieval vs majority: "
    f"{(accuracy - 0.4867):+.2%}"
)

print(
    f"V3 vs retrieval: "
    f"{(0.6133 - accuracy):+.2%}"
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

output_path = (
    "evaluation/retrieval_baseline_results.csv"
)

golden.to_csv(
    output_path,
    index=False
)

print(f"\nSaved: {output_path}")