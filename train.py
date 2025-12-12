# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

INPUT_CSV = "prepared_data.csv"
METRICS_CSV = "metrics.csv"
PREDICTIONS_FILE = "predictions.npz"

def evaluate_model(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

def main():
    df = pd.read_csv(INPUT_CSV)

    X = df.drop(columns=["target_is_movie"]).values
    y = df["target_is_movie"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GaussianNB": GaussianNB(),
    }

    metrics = {}
    preds = {}
    probas = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics[name] = evaluate_model(y_test, y_pred, y_proba)
        preds[name] = y_pred
        probas[name] = y_proba

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(METRICS_CSV, index=True)
    print(f"Metrics saved to {METRICS_CSV}")

    np.savez(
        PREDICTIONS_FILE,
        y_test=y_test,
        **{f"{name}_pred": arr for name, arr in preds.items()},
        **{f"{name}_proba": arr for name, arr in probas.items()},
    )
    print(f"Predictions saved to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()
