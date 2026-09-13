import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/apple_conversations.csv"


class HistoricalRetriever:

    def __init__(self):
        print("Loading AppleSupport conversations...")

        self.data = pd.read_csv(DATA_PATH)

        self.data["customer_text"] = (
            self.data["customer_text"]
            .fillna("")
            .astype(str)
        )

        self.data["support_text"] = (
            self.data["support_text"]
            .fillna("")
            .astype(str)
        )

        print(f"Loaded {len(self.data)} conversations")

        # Convert historical customer messages into TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )

        print("Building search index...")

        self.customer_vectors = self.vectorizer.fit_transform(
            self.data["customer_text"]
        )

        print("Search index ready!")

    def search(self, query, top_k=5):

        # Convert new customer message into TF-IDF
        query_vector = self.vectorizer.transform([query])

        # Calculate similarity
        scores = cosine_similarity(
            query_vector,
            self.customer_vectors
        )[0]

        # Get highest scoring conversations
        top_indices = scores.argsort()[-top_k:][::-1]

        results = self.data.iloc[top_indices].copy()

        results["similarity"] = scores[top_indices]

        return results[
            [
                "customer_tweet_id",
                "customer_text",
                "support_tweet_id",
                "support_text",
                "similarity"
            ]
        ]


if __name__ == "__main__":

    retriever = HistoricalRetriever()

    query = "My iPhone battery is draining very quickly"

    results = retriever.search(query, top_k=5)

    print("\nQuery:")
    print(query)

    print("\nMost similar historical conversations:\n")

    for _, row in results.iterrows():

        print("-" * 70)

        print("Customer:")
        print(row["customer_text"])

        print("\nAppleSupport:")
        print(row["support_text"])

        print(f"\nSimilarity: {row['similarity']:.3f}")