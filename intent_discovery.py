import pandas as pd
from pathlib import Path
import re

INPUT_FILE = Path("data/apple_conversations.csv")

print("Loading AppleSupport conversations...")

df = pd.read_csv(INPUT_FILE)

print(f"Total conversations: {len(df):,}")

# Basic text cleaning
df["customer_text"] = df["customer_text"].fillna("")

def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["clean_text"] = df["customer_text"].apply(clean_text)

# Common keywords for initial exploration
keywords = {
    "battery": [
        "battery", "drain", "charging", "charge", "charger"
    ],
    "ios_update": [
        "ios", "update", "upgrade", "ios 11", "ios 12", "ios 13"
    ],
    "app_problem": [
        "app", "apps", "crash", "crashing", "freeze", "freezing"
    ],
    "apple_music": [
        "apple music", "music", "itunes"
    ],
    "account_login": [
        "login", "log in", "sign in", "password", "account",
        "verification", "code", "icloud"
    ],
    "device_problem": [
        "iphone", "ipad", "mac", "device", "phone", "keyboard",
        "screen", "camera"
    ],
    "purchase_order": [
        "order", "purchase", "shipping", "delivery", "refund",
        "return", "reserved"
    ],
    "connectivity": [
        "wifi", "wi-fi", "internet", "network", "bluetooth",
        "connection", "connect"
    ],
    "notification": [
        "notification", "notifications", "alert"
    ],
    "support_contact": [
        "help", "support", "call", "phone number", "contact",
        "customer service", "call centre", "call center"
    ]
}

print("\nKeyword-based issue distribution:")
print("=" * 60)

results = []

for category, words in keywords.items():

    pattern = "|".join(
        re.escape(word) for word in words
    )

    matches = df["clean_text"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    )

    count = matches.sum()

    results.append({
        "category": category,
        "matches": count,
        "percentage": round(count / len(df) * 100, 2)
    })

results_df = (
    pd.DataFrame(results)
    .sort_values("matches", ascending=False)
)

print(results_df.to_string(index=False))

results_df.to_csv(
    "data/initial_intent_analysis.csv",
    index=False
)

print("\n" + "=" * 60)
print("Sample messages by category")
print("=" * 60)

for category, words in keywords.items():

    pattern = "|".join(
        re.escape(word) for word in words
    )

    matches = df[
        df["clean_text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ]

    print(f"\n### {category.upper()}")

    for text in matches["customer_text"].head(3):
        print("-", text)

print("\nSaved:")
print("data/initial_intent_analysis.csv")