import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_PATH = "hogwarts_legacy_reviews.csv"
MODEL_PATH = "sentiment_pipeline.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df[["Review", "Feedback"]].dropna()
    df["Review"] = df["Review"].astype(str)
    df["Feedback"] = df["Feedback"].astype(str)
    df = df[df["Feedback"].isin(["Positive", "Negative"])]

    X = df["Review"]
    y = df["Feedback"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            random_state=42,
            class_weight="balanced"  # imbalance fix
        ))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print("\nMacro F1:", f1_score(y_test, preds, average="macro"))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nReport:\n", classification_report(y_test, preds))

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()