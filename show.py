# show.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

METRICS_CSV = "metrics.csv"
PREDICTIONS_FILE = "predictions.npz"
OUT_DIR = "plots"

def main():
    metrics_df = pd.read_csv(METRICS_CSV, index_col=0)
    print("=== Metrics ===")
    print(metrics_df)

    data = np.load(PREDICTIONS_FILE)
    y_test = data["y_test"]

    model_names = ["DecisionTree", "KNN", "LogisticRegression", "GaussianNB"]

    os.makedirs(OUT_DIR, exist_ok=True)

    for name in model_names:
        y_pred = data[f"{name}_pred"]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(name)
        plt.tight_layout()

        # автоимя: plots/confusion_matrix_DecisionTree.png и т.д.
        out_path = os.path.join(OUT_DIR, f"confusion_matrix_{name}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
