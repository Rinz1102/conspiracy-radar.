"""
train_model.py
Trains the TF-IDF + Logistic Regression classifier on the labeled
starter dataset and saves it to model.joblib.

Run this whenever you expand data/labeled_texts.csv with more examples.

Usage:
    python train_model.py
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

DATA_PATH = Path(__file__).parent / "data" / "labeled_texts.csv"
MODEL_PATH = Path(__file__).parent / "model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} labeled examples "
          f"({(df.label == 1).sum()} conspiracy-style, "
          f"{(df.label == 0).sum()} neutral)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    cv_scores = cross_val_score(pipeline, df["text"], df["label"], cv=5)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Refit on the FULL dataset before saving, so the shipped model
    # benefits from every labeled example, not just the training split.
    pipeline.fit(df["text"], df["label"])
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
