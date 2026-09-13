import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from retrieval import HistoricalRetriever
from generate_reply import generate_reply
from escalation import decide_escalation


MODEL_PATH = Path("models/intent_classifier.pkl")
GOLDEN_PATH = Path("data/golden_set.csv")
OUTPUT_PATH = Path("evaluation/reply_evaluation_results.csv")


def main():
    golden = pd.read_csv(GOLDEN_PATH)

    print(f"Golden examples: {len(golden)}")

    # Load classifier
    model = joblib.load(MODEL_PATH)
    print("Classifier loaded.")

    # Load historical retriever
    retriever = HistoricalRetriever()

    # Remove golden examples from retrieval to prevent leakage
    golden_ids = set(
        golden["customer_tweet_id"]
        .dropna()
        .astype(str)
    )

    retriever.data["customer_tweet_id"] = (
        retriever.data["customer_tweet_id"].astype(str)
    )

    retriever.data = retriever.data[
        ~retriever.data["customer_tweet_id"].isin(golden_ids)
    ].reset_index(drop=True)

    print(f"Removed {len(golden_ids)} golden examples from retrieval.")

    # Rebuild search index after removing golden examples
    print("Rebuilding search index...")

    retriever.customer_vectors = retriever.vectorizer.transform(
        retriever.data["customer_text"]
    )

    print("Leakage-free search index ready!")

    results = []

    for index, row in golden.iterrows():

        customer_text = str(row["customer_text"])

        # Predict intent
        predicted_intent = model.predict([customer_text])[0]

        # Retrieve historical evidence
        retrieved = retriever.search(customer_text, top_k=3)

        # Generate reply
        draft_reply = generate_reply(
            customer_text,
            predicted_intent,
            retrieved
        )

        # Escalation decision
        escalation = decide_escalation(
            predicted_intent,
            retrieved
        )

        record = {
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": customer_text,
            "expected_intent": row["intent"],
            "predicted_intent": predicted_intent,
            "draft_reply": draft_reply,
            "decision": escalation["decision"],
            "decision_reason": escalation["reason"],
        }

        # Save top-3 ACTUAL historical evidence
        for i in range(3):

            if i < len(retrieved):

                evidence = retrieved.iloc[i]

                record[f"evidence_{i+1}_customer"] = str(
                    evidence["customer_text"]
                )

                record[f"evidence_{i+1}_support"] = str(
                    evidence["support_text"]
                )

                record[f"evidence_{i+1}_similarity"] = float(
                    evidence["similarity"]
                )

            else:

                record[f"evidence_{i+1}_customer"] = ""

                record[f"evidence_{i+1}_support"] = ""

                record[f"evidence_{i+1}_similarity"] = 0.0

        results.append(record)

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{len(golden)}")

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Evaluation complete.")
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("LEAKAGE-FREE REPLY EVALUATION")

    print(
        f"Intent accuracy: "
        f"{(output['expected_intent'] == output['predicted_intent']).mean():.2%}"
    )

    print(
        f"Non-empty replies: "
        f"{(output['draft_reply'].str.strip() != '').mean():.2%}"
    )

    print(
        f"Evidence found: "
        f"{(output['evidence_1_support'].str.strip() != '').mean():.2%}"
    )

    print(
        f"Auto-handle rate: "
        f"{(output['decision'] == 'AUTO-HANDLE').mean():.2%}"
    )

    print(
        f"Escalation rate: "
        f"{(output['decision'] == 'ESCALATE').mean():.2%}"
    )

    print(
        f"Average top similarity: "
        f"{output['evidence_1_similarity'].mean():.3f}"
    )


if __name__ == "__main__":
    main()