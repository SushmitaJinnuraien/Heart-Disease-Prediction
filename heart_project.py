"""Heart disease prediction: data analysis (Problem 1) and model comparison (Problem 2).

Usage:
    python heart_project.py                      # required analysis + models
    python heart_project.py --extras             # also repeated cross-validation
    python heart_project.py --data path/to/heart1.csv --out results
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/heart1.csv", help="path to heart1.csv")
    p.add_argument("--out", default="results", help="folder for figures and tables")
    p.add_argument("--extras", action="store_true",
                   help="also run repeated cross-validation (robustness check)")
    return p.parse_args()


def problem1(df, out):
    print("Shape:", df.shape, "| missing values:", int(df.isna().sum().sum()))
    print("\nSummary statistics:\n", df.describe().T.round(2))
    print("\nClass counts (1 = absence, 2 = presence):\n", df["a1p2"].value_counts())

    corr = df.corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nCorrelation of each variable with a1p2 (heart disease):")
    print(corr["a1p2"].drop("a1p2").sort_values(key=abs, ascending=False).round(3))

    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
                 .stack().rename("r").reset_index())
    pairs.columns = ["var1", "var2", "r"]
    print("\nTop 8 correlated pairs:")
    print(pairs.reindex(pairs.r.abs().sort_values(ascending=False).index)
               .head(8).round(3))

    cov = df.cov()
    print("\nCovariance matrix:\n", cov.round(2))
    cov.to_csv(out / "covariance_matrix.csv")
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.apply(lambda c: (c - c.mean()) / c.std()).cov(), annot=True,
                fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Covariance Matrix (standardized variables)")
    plt.tight_layout()
    plt.savefig(out / "covariance_heatmap.png", dpi=150)
    plt.close()

    sns.pairplot(df, hue="a1p2", palette={1: "tab:blue", 2: "tab:orange"},
                 plot_kws={"s": 12, "alpha": 0.6}, corner=True)
    plt.savefig(out / "pairplot.png", dpi=100)
    plt.close()


def problem2(df, out, extras):
    X = df.drop(columns="a1p2").values          # all 13 features, no PCA
    y = df["a1p2"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    sc = StandardScaler().fit(X_train)           # fit on training data only
    X_train_std, X_test_std = sc.transform(X_train), sc.transform(X_test)

    # Best parameters, found beforehand by 5-fold CV on the training set
    models = {
        "Perceptron": Perceptron(eta0=0.1, max_iter=10, penalty="l2",
                                 alpha=1e-4, random_state=1),
        "Logistic Regression": LogisticRegression(C=0.03, max_iter=5000),
        "SVM (linear)": SVC(kernel="linear", C=0.01, probability=True,
                            random_state=1),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=4,
                                                random_state=1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1),
        "K-Nearest Neighbor (K=30)": KNeighborsClassifier(n_neighbors=30),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train_std, y_train)
        rows.append({
            "Method": name,
            "Train Accuracy (%)": round(100 * accuracy_score(
                y_train, model.predict(X_train_std)), 1),
            "Test Accuracy (%)": round(100 * accuracy_score(
                y_test, model.predict(X_test_std)), 1)})
        if hasattr(model, "predict_proba"):      # Perceptron has none
            rows[-1]["Mean Confidence (%)"] = round(
                100 * model.predict_proba(X_test_std).max(axis=1).mean(), 1)

    results = pd.DataFrame(rows).fillna("n/a")
    print("\nModel comparison:\n", results.to_string(index=False))
    results.to_csv(out / "model_comparison.csv", index=False)

    best = models["Logistic Regression"]
    for i, p in enumerate(best.predict_proba(X_test_std[:3])):
        print(f"Patient {i}: predicted class {best.classes_[p.argmax()]} "
              f"(confidence {100 * p.max():.1f}%), actual {y_test[i]}")

    # Confusion counts (positive class = 2 = heart disease)
    cm_rows = []
    for name, model in models.items():
        pred = model.predict(X_test_std)
        # labels=[1, 2]: class 1 = no disease (negative), class 2 = disease (positive)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[1, 2]).ravel()
        cm_rows.append({"Method": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                        "Sensitivity (%)": round(100 * tp / (tp + fn), 1),
                        "Specificity (%)": round(100 * tn / (tn + fp), 1),
                        "Precision (%)": round(100 * tp / (tp + fp), 1)})
    cm = pd.DataFrame(cm_rows)
    print("\nConfusion matrix counts (positive = heart disease):\n",
          cm.to_string(index=False))
    cm.to_csv(out / "confusion_counts.csv", index=False)

    if extras:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
        cv_rows = []
        for name, model in models.items():
            s = cross_val_score(make_pipeline(StandardScaler(), model), X, y, cv=cv)
            cv_rows.append({"Method": name,
                            "CV Mean Accuracy (%)": round(100 * s.mean(), 1),
                            "CV Std (%)": round(100 * s.std(), 1)})
        cv_t = pd.DataFrame(cv_rows)
        print("\nRepeated 5-fold CV (10 repeats):\n", cv_t.to_string(index=False))
        cv_t.to_csv(out / "cv_comparison.csv", index=False)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    problem1(df, out)
    problem2(df, out, args.extras)


if __name__ == "__main__":
    main()
