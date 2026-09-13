def decide_escalation(intent, results):
    """
    Decide whether a customer request should be auto-handled
    or escalated to a human agent.
    """

    # No historical evidence
    if results.empty:
        return {
            "decision": "ESCALATE",
            "reason": "No historical AppleSupport evidence was retrieved."
        }

    # Best retrieval similarity
    best_similarity = float(results.iloc[0]["similarity"])

    # Sensitive / account-related issues
    if intent == "account_icloud":
        return {
            "decision": "ESCALATE",
            "reason": "Account and iCloud issues may require account-specific verification."
        }

    # Purchase/refund/order issues
    if intent == "order_purchase":
        return {
            "decision": "ESCALATE",
            "reason": "Purchase, refund, and order issues may require transaction-specific support."
        }

    # Weak historical evidence
    if best_similarity < 0.25:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Historical evidence is weak "
                f"(similarity={best_similarity:.3f})."
            )
        }

    # Strong evidence for routine support issues
    if best_similarity >= 0.45:
        return {
            "decision": "AUTO-HANDLE",
            "reason": (
                f"Strong historical evidence was found "
                f"(similarity={best_similarity:.3f}) "
                f"for this support issue."
            )
        }

    # Medium confidence → human review
    return {
        "decision": "ESCALATE",
        "reason": (
            f"Historical evidence is only moderate "
            f"(similarity={best_similarity:.3f}); "
            "human review is safer."
        )
    }


if __name__ == "__main__":
    print("Escalation module ready.")