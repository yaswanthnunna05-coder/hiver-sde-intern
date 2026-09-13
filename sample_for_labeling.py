import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/apple_conversations.csv")
OUTPUT_FILE = Path("data/labeling_candidates.csv")

df = pd.read_csv(INPUT_FILE)

# Clean basic issues
df = df.dropna(subset=["customer_text"])
df = df.drop_duplicates(subset=["customer_text"])

# Random sample
sample = df.sample(
    n=min(100, len(df)),
    random_state=42
).copy()

# Add empty label columns
sample["intent"] = ""
sample["notes"] = ""

# Keep only what we need for manual labeling
sample = sample[
    [
        "customer_tweet_id",
        "customer_text",
        "support_text",
        "intent",
        "notes"
    ]
]

sample.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 60)
print("LABELING SAMPLE CREATED")
print("=" * 60)

print(f"Examples: {len(sample)}")
print(f"File: {OUTPUT_FILE}")

print("\nIntent labels to use:")
print("""
ios_update
device_issue
app_issue
battery_charging
account_icloud
connectivity
music_media
order_purchase
other_support
""")

print("\nOpen this file in VS Code/Excel and manually label the")
print("intent column.")