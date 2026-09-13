import sys
from pathlib import Path

import joblib

# Allow importing retrieval.py
sys.path.append(str(Path(__file__).resolve().parent))

from retrieval import HistoricalRetriever


MODEL_PATH = Path("models/intent_classifier.pkl")


def load_classifier():
    return joblib.load(MODEL_PATH)


def predict_intent(model, customer_text):
    return model.predict([customer_text])[0]


def clean_support_text(text):
    return " ".join(str(text).split())


def generate_reply(customer_text, intent, results):

    if results.empty:
        return (
            "Thanks for reaching out. We'd like to help with this issue. "
            "Please contact Apple Support so we can look into it further."
        )

    # Best historical response
    historical_reply = clean_support_text(
        results.iloc[0]["support_text"]
    )

    reply_lower = historical_reply.lower()

    if "dm" in reply_lower or "direct message" in reply_lower:
        return (
            "Sorry you're having trouble with this. "
            "We'd be happy to look into it further. "
            "Please send us a DM so we can get more details and help you."
        )

    if "update" in reply_lower:
        return (
            "Sorry you're experiencing this issue. "
            "Please make sure your device is running the latest available "
            "software update. If the issue continues, please contact "
            "Apple Support with more details."
        )

    if "restart" in reply_lower:
        return (
            "Sorry you're experiencing this issue. "
            "A restart may help resolve the problem. "
            "If the issue continues, please contact Apple Support "
            "with more details."
        )

    if "settings" in reply_lower:
        return (
            "Sorry you're experiencing this issue. "
            "Please check the relevant settings on your device. "
            "If the problem continues, please contact Apple Support "
            "so we can look into it further."
        )

    return (
        "Sorry you're experiencing this issue. "
        "Based on similar Apple Support cases, we'd recommend "
        "continuing with Apple Support so the issue can be "
        "investigated further."
    )


def main():

    customer_text = input("\nCustomer message: ").strip()

    if not customer_text:
        print("Please enter a customer message.")
        return

    print("\nLoading classifier...")
    model = load_classifier()

    intent = predict_intent(model, customer_text)

    print(f"Predicted intent: {intent}")

    print("\nLoading historical retrieval system...")
    retriever = HistoricalRetriever()

    print("\nRetrieving similar historical cases...")
    results = retriever.search(customer_text, top_k=5)

    print(f"Retrieved {len(results)} historical cases.")

    reply = generate_reply(
        customer_text,
        intent,
        results
    )

    print("\n--- DRAFT REPLY ---")
    print(reply)

    print("\n--- EVIDENCE ---")

    for i, (_, row) in enumerate(results.head(3).iterrows(), start=1):

        print(f"\n{i}. Similarity: {row['similarity']:.3f}")

        print(f"Customer: {row['customer_text']}")

        print(f"AppleSupport: {row['support_text']}")


if __name__ == "__main__":
    main()