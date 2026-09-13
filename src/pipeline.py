import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).resolve().parent))

from retrieval import HistoricalRetriever
from generate_reply import generate_reply
from escalation import decide_escalation


MODEL_PATH = Path("models/intent_classifier.pkl")


def load_classifier():
    return joblib.load(MODEL_PATH)


def run_agent(customer_text, model, retriever):
    # 1. Predict intent
    intent = model.predict([customer_text])[0]

    # 2. Retrieve similar historical cases
    results = retriever.search(customer_text, top_k=5)

    # 3. Generate grounded reply
    reply = generate_reply(customer_text, intent, results)

    # 4. Decide auto-handle or escalate
    escalation = decide_escalation(intent, results)

    return {
        "intent": intent,
        "reply": reply,
        "decision": escalation["decision"],
        "reason": escalation["reason"],
        "evidence": results
    }


def main():
    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 70)

    customer_text = input("\nCustomer message: ").strip()

    if not customer_text:
        print("Please enter a customer message.")
        return

    print("\nLoading intent classifier...")
    model = load_classifier()

    print("Loading historical retrieval system...")
    retriever = HistoricalRetriever()

    print("\nRunning support agent...")

    result = run_agent(customer_text, model, retriever)

    print("\n" + "=" * 70)
    print("AGENT RESULT")
    print("=" * 70)

    print("\nIntent:")
    print(result["intent"])

    print("\nDraft Reply:")
    print(result["reply"])

    print("\nDecision:")
    print(result["decision"])

    print("\nReason:")
    print(result["reason"])

    print("\n" + "-" * 70)
    print("TOP HISTORICAL EVIDENCE")
    print("-" * 70)

    for i, (_, row) in enumerate(
        result["evidence"].head(3).iterrows(),
        start=1
    ):
        print(f"\nEvidence {i}")
        print(f"Similarity: {row['similarity']:.3f}")
        print(f"Customer: {row['customer_text']}")
        print(f"AppleSupport: {row['support_text']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()