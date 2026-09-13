from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# Find CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found. Put the Kaggle dataset CSV inside data/raw/"
    )

csv_path = csv_files[0]

print(f"Using dataset: {csv_path}")
print("=" * 60)

# Required columns
usecols = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

brand_counts = {}

# Read dataset in chunks so we don't load the entire dataset into memory
for chunk in pd.read_csv(
    csv_path,
    usecols=usecols,
    dtype="string",
    chunksize=200_000
):
    chunk["inbound"] = chunk["inbound"].str.upper()

    # Company responses have inbound = FALSE
    company_rows = chunk[chunk["inbound"] == "FALSE"]

    counts = company_rows["author_id"].value_counts()

    for brand, count in counts.items():
        brand_counts[brand] = brand_counts.get(brand, 0) + int(count)

# Convert to DataFrame
brands = (
    pd.DataFrame(
        list(brand_counts.items()),
        columns=["brand", "company_responses"]
    )
    .sort_values("company_responses", ascending=False)
)

print("\nTop brands/accounts in the dataset:")
print("=" * 60)
print(brands.head(25).to_string(index=False))

# Save results
brands.to_csv("data/brand_counts.csv", index=False)

print("\nSaved results to: data/brand_counts.csv")