from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/raw/twcs.csv")
OUTPUT_FILE = Path("data/apple_support.csv")

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

print("Reading AppleSupport data...")

apple_chunks = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=USECOLS,
    dtype="string",
    chunksize=200_000
):
    chunk["inbound"] = chunk["inbound"].str.upper()

    # AppleSupport company responses
    company = chunk[
        (chunk["inbound"] == "FALSE") &
        (chunk["author_id"] == "AppleSupport")
    ]

    # Customer tweets directed to AppleSupport
    customer = chunk[
        (chunk["inbound"] == "TRUE")
    ]

    apple_chunks.append(company)

    print(
        f"Found {len(company):,} AppleSupport responses in this chunk"
    )

apple = pd.concat(apple_chunks, ignore_index=True)

# Remove empty tweets
apple = apple.dropna(subset=["text"])

# Remove duplicates
apple = apple.drop_duplicates(subset=["tweet_id"])

print("\n" + "=" * 60)
print("APPLE SUPPORT DATA")
print("=" * 60)

print(f"Total AppleSupport responses: {len(apple):,}")

print("\nSample conversations:")
print("-" * 60)

for _, row in apple.head(10).iterrows():
    print(f"\nTweet ID: {row['tweet_id']}")
    print(f"Text: {row['text']}")
    print(f"Response to: {row['in_response_to_tweet_id']}")

# Save
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
apple.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 60)