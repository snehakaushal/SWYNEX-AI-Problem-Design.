import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

DATA_FILE = "support_tickets.csv"
MODEL_FILE = "support_ticket_model.pkl"

def build_model():
    text_features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        ))
    ])

    return Pipeline([
        ("features", text_features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
            random_state=42
        ))
    ])

def main():
    parser = argparse.ArgumentParser(description="Support ticket category classifier")
    parser.add_argument("--message", help="Optional ticket message to classify")
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["message", "category"]).drop_duplicates(subset=["message"])
    df["message"] = df["message"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["message"] != "") & (df["category"] != "")]

    print(f"Dataset rows: {len(df)}")
    print("\nExamples per category:")
    print(df["category"].value_counts())

    X = df["message"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTest Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Test Macro F1:", round(f1_score(y_test, predictions, average="macro"), 3))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion Matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=model.classes_),
        index=model.classes_,
        columns=model.classes_
    ))

    # Fit the final model on all available labeled examples for future predictions.
    final_model = build_model()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_FILE)
    print(f"\nSaved final model to: {MODEL_FILE}")

    message = args.message
    if not message:
        message = input("\nEnter a support ticket to classify (or press Enter to exit): ").strip()

    if message:
        predicted_category = final_model.predict([message])[0]
        probabilities = final_model.predict_proba([message])[0]
        confidence = max(probabilities)

        print("\nPredicted category:", predicted_category)
        print("Model confidence:", f"{confidence:.1%}")
        print("\nScores by category:")
        for category, score in sorted(
            zip(final_model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True
        ):
            print(f"  {category}: {score:.1%}")

        if confidence < 0.50:
            print("\nThis prediction is uncertain. Consider manual review.")

if __name__ == "__main__":
    main()
