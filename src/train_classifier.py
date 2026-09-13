import os
import re
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


DATA_PATH = "data/apple_conversations.csv"
GOLDEN_PATH = "data/golden_set.csv"
MODEL_PATH = "models/intent_classifier.pkl"


# --------------------------------------------------
# 1. Assign weak training labels from historical data
# --------------------------------------------------

def assign_intent(text):
    text = str(text).lower()

    # --------------------------------------------------
    # 1. Account / iCloud
    # --------------------------------------------------
    if any(x in text for x in [
        "apple id", "icloud", "forgot password",
        "verification code", "two factor", "2fa",
        "sign in", "login", "password"
    ]):
        return "account_icloud"

    # --------------------------------------------------
    # 2. Orders / purchases
    # --------------------------------------------------
    if any(x in text for x in [
        "order", "purchase", "refund", "delivery",
        "shipping", "bought", "buy"
    ]):
        return "order_purchase"

    # --------------------------------------------------
    # 3. Battery
    # --------------------------------------------------
    if any(x in text for x in [
        "battery",
        "battery life",
        "battery drain",
        "battery draining",
        "draining quickly",
        "draining fast",
        "won't hold charge",
        "won't keep a charge",
        "not holding charge",
        "charge won't",
        "charging"
    ]):
        return "battery_charging"

    # --------------------------------------------------
    # 4. Connectivity
    # --------------------------------------------------
    if any(x in text for x in [
        "wifi", "wi-fi", "bluetooth",
        "internet", "network",
        "connection", "connect"
    ]):
        return "connectivity"

    # --------------------------------------------------
    # 5. Music / media
    # --------------------------------------------------
    if any(x in text for x in [
        "apple music",
        "itunes music",
        "playlist",
        "song",
        "album",
        "music",
        "carplay playlist"
    ]):
        return "music_media"

    # --------------------------------------------------
    # 6. App-specific problems
    # --------------------------------------------------
    if any(x in text for x in [
        "app crash",
        "app crashing",
        "app freezes",
        "app freezing",
        "app won't open",
        "app won't work",
        "application",
        "instagram",
        "facebook",
        "twitter"
    ]):
        return "app_issue"

    # --------------------------------------------------
    # 7. Explicit iOS update requests/problems
    # --------------------------------------------------
    update_problem = any(x in text for x in [
        "update won't install",
        "update will not install",
        "cannot update",
        "can't update",
        "cant update",
        "unable to update",
        "failed to update",
        "update failed",
        "update error",
        "can't download update",
        "cannot download update",
        "how to update",
        "how do i update",
        "can i update",
        "upgrade",
        "downgrade"
    ])

    ios_version = any(x in text for x in [
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "ios 19"
    ])

    # If the update itself is the problem, use ios_update.
    if update_problem:
        return "ios_update"

    # Explicit iOS-version discussion without a clearer
    # device/app/battery/connectivity problem.
    if ios_version:
        if not any(x in text for x in [
            "battery",
            "charging",
            "wifi",
            "wi-fi",
            "bluetooth",
            "instagram",
            "facebook",
            "twitter",
            "app",
            "application",
            "crash",
            "crashing",
            "freezing",
            "freezes",
            "glitch",
            "slow",
            "lag",
            "keyboard",
            "screen",
            "camera",
            "speaker",
            "touch",
            "notification",
            "emoji",
            "autocorrect"
        ]):
            return "ios_update"

    # --------------------------------------------------
    # 8. Device problems
    # --------------------------------------------------
    if any(x in text for x in [
        "iphone",
        "ipad",
        "mac",
        "keyboard",
        "screen",
        "camera",
        "speaker",
        "phone",
        "touch",
        "restart",
        "reboot",
        "notification",
        "brightness",
        "storage",
        "emoji",
        "autocorrect",
        "glitch",
        "freezes",
        "freezing",
        "crashing",
        "slow",
        "lag",
        "won't turn on",
        "doesn't work",
        "does not work"
    ]):
        return "device_issue"

    # --------------------------------------------------
    # 9. General update wording
    # --------------------------------------------------
    if any(x in text for x in [
        "update",
        "ios",
        "software update"
    ]):
        return "ios_update"

    # --------------------------------------------------
    # 10. Other
    # --------------------------------------------------
    return "other_support"
# --------------------------------------------------
# 2. Load historical AppleSupport conversations
# --------------------------------------------------

print("Loading historical conversations...")

df = pd.read_csv(DATA_PATH)

df["customer_text"] = df["customer_text"].fillna("").astype(str)

print("Historical conversations:", len(df))


# --------------------------------------------------
# 3. Create weak labels for training
# --------------------------------------------------

df["intent"] = df["customer_text"].apply(assign_intent)

print("\nTraining intent distribution:")
print(df["intent"].value_counts())


# --------------------------------------------------
# 4. Remove golden examples from training
# --------------------------------------------------

golden = pd.read_csv(GOLDEN_PATH)

golden_ids = set(
    golden["customer_tweet_id"]
    .astype(str)
)

df["id_string"] = df["customer_tweet_id"].astype(str)

train_df = df[~df["id_string"].isin(golden_ids)].copy()

print("\nGolden examples excluded from training:", len(df) - len(train_df))
print("Training examples:", len(train_df))


# --------------------------------------------------
# 5. Train TF-IDF + Logistic Regression
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

print("\nTraining classifier...")

model.fit(
    train_df["customer_text"],
    train_df["intent"]
)


# --------------------------------------------------
# 6. Save model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)

print("\nClassifier training complete!")