import pandas as pd

INPUT_PATH = "evaluation/reply_evaluation_results.csv"
OUTPUT_PATH = "evaluation/human_evidence_annotation.csv"

df = pd.read_csv(INPUT_PATH)

annotation = df[
    [
        "customer_tweet_id",
        "customer_text",
        "expected_intent",
        "predicted_intent",
        "draft_reply",
        "decision",
        "evidence_1_customer",
        "evidence_1_support",
        "evidence_1_similarity",
        "evidence_2_customer",
        "evidence_2_support",
        "evidence_2_similarity",
        "evidence_3_customer",
        "evidence_3_support",
        "evidence_3_similarity",
    ]
].copy()

annotation["human_evidence_supported"] = ""
annotation["human_notes"] = ""

annotation.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 60)
print("HUMAN EVIDENCE ANNOTATION FILE")
print("=" * 60)

print(f"Examples: {len(annotation)}")
print(f"Saved: {OUTPUT_PATH}")
print()
print("Open this CSV in Excel and label:")
print("human_evidence_supported")
print()
print("Use:")
print("1 = Evidence supports the draft reply")
print("0 = Evidence does not support the draft reply")