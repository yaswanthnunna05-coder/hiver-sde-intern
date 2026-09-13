from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/raw/twcs.csv")
OUTPUT_FILE = Path("data/apple_conversations.csv")

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

print("Step 1: Finding AppleSupport responses...")

apple_responses = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=USECOLS,
    dtype="string",
    chunksize=200_000
):
    chunk["inbound"] = chunk["inbound"].str.upper()

    rows = chunk[
        (chunk["inbound"] == "FALSE") &
        (chunk["author_id"] == "AppleSupport")
    ]

    apple_responses.append(rows)

apple_responses = pd.concat(
    apple_responses,
    ignore_index=True
)

apple_responses = apple_responses.dropna(
    subset=["in_response_to_tweet_id", "text"]
)

print(f"AppleSupport responses found: {len(apple_responses):,}")

# IDs of customer tweets that AppleSupport replied to
customer_ids = set(
    apple_responses["in_response_to_tweet_id"]
    .dropna()
    .tolist()
)

print(f"Customer tweets to find: {len(customer_ids):,}")

print("\nStep 2: Finding the corresponding customer tweets...")

customer_tweets = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=USECOLS,
    dtype="string",
    chunksize=200_000
):
    chunk["inbound"] = chunk["inbound"].str.upper()

    rows = chunk[
        (chunk["inbound"] == "TRUE") &
        (chunk["tweet_id"].isin(customer_ids))
    ]

    if len(rows) > 0:
        customer_tweets.append(rows)

customer_tweets = pd.concat(
    customer_tweets,
    ignore_index=True
)

print(f"Customer tweets matched: {len(customer_tweets):,}")

# Create lookup: customer tweet ID -> customer text
customer_lookup = (
    customer_tweets
    .drop_duplicates("tweet_id")
    .set_index("tweet_id")["text"]
    .to_dict()
)

# Build customer -> AppleSupport response pairs
pairs = []

for _, row in apple_responses.iterrows():

    customer_id = row["in_response_to_tweet_id"]

    if customer_id not in customer_lookup:
        continue

    pairs.append({
        "customer_tweet_id": customer_id,
        "customer_text": customer_lookup[customer_id],
        "support_tweet_id": row["tweet_id"],
        "support_text": row["text"],
        "created_at": row["created_at"]
    })

conversations = pd.DataFrame(pairs)

# Remove empty messages
conversations = conversations.dropna(
    subset=["customer_text", "support_text"]
)

# Remove duplicate pairs
conversations = conversations.drop_duplicates(
    subset=["customer_tweet_id", "support_tweet_id"]
)

print("\n" + "=" * 60)
print("RECONSTRUCTED CONVERSATIONS")
print("=" * 60)

print(f"Total customer → AppleSupport pairs: {len(conversations):,}")

print("\nSample conversations:")
print("-" * 60)

for _, row in conversations.head(15).iterrows():

    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("\nAPPLE SUPPORT:")
    print(row["support_text"])

    print("-" * 60)

# Save
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

conversations.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved to: {OUTPUT_FILE}")
