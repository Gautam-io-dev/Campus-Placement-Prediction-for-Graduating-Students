"""
Campus-Placement Prediction for Graduating Students
Student: Gautam Mishra | Roll No.: 24020570012 | Problem No.: 19

Run:
    pip install pandas numpy scikit-learn matplotlib
    python campus_placement_experiments.py --data Placement_Data_Full_Class.csv

This script computes the results from the real CSV supplied by the student.
It deliberately does not hard-code or fabricate performance scores.
"""
import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import make_column_selector

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Placement_Data_Full_Class.csv",
                        help="Path to Placement_Data_Full_Class.csv")
    parser.add_argument("--outdir", default="results",
                        help="Directory for tables and figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"status", "salary"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain 'status' and 'salary' columns.")
    if "sl_no" in df.columns:
        df = df.drop(columns=["sl_no"])  # identifier, not a predictive feature

    print("\\n=== DATA AUDIT ===")
    print("Shape:", df.shape)
    print("\\nMissing values:\\n", df.isna().sum().to_string())
    print("\\nPlacement counts:\\n", df["status"].value_counts(dropna=False).to_string())

    # Classification: salary is excluded because it is only known after placement.
    X = df.drop(columns=["status", "salary"], errors="ignore")
    y = df["status"].astype(str).str.strip().str.lower().map(
        lambda s: 1 if s == "placed" else 0
    )
    if y.isna().any():
        raise ValueError("Unexpected values in status column; inspect CSV.")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42,
                                                  class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB()
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        # GaussianNB needs dense input; the transformed matrix is small here.
        if name == "Gaussian Naive Bayes":
            from sklearn.preprocessing import FunctionTransformer
            pipe = Pipeline([
                ("prep", preprocess),
                ("dense", FunctionTransformer(lambda a: a.toarray() if hasattr(a, "toarray") else a)),
                ("model", model)
            ])
        else:
            pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if proba is not None and len(np.unique(y_test)) == 2 else np.nan
        }
        rows.append(row)
        fitted[name] = pipe
        print(f"\\n=== {name} ===")
        print(pd.Series(row).to_string())
        print("Confusion matrix [[TN, FP], [FN, TP]]:\\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, target_names=["Not Placed", "Placed"], zero_division=0))

    cls = pd.DataFrame(rows).sort_values("f1", ascending=False)
    cls.to_csv(os.path.join(args.outdir, "classification_metrics.csv"), index=False)
    print("\\n=== CLASSIFICATION COMPARISON ===\\n", cls.to_string(index=False))

    # Cross-validation: preprocessing remains inside the pipeline to avoid leakage.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    scoring = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    for name, pipe in fitted.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")
        cv_rows.append({"model": name, **{
            f"{metric}_cv_mean": float(np.mean(scores[f"test_{metric}"]))
            for metric in scoring
        }})
    cv_df = pd.DataFrame(cv_rows).sort_values("f1_cv_mean", ascending=False)
    cv_df.to_csv(os.path.join(args.outdir, "classification_cv_metrics.csv"), index=False)
    print("\\n=== 5-FOLD STRATIFIED CV ===\\n", cv_df.to_string(index=False))

    # Salary regression: only records with an observed salary are used.
    # This is conditional salary estimation among placed candidates, not all students.
    salary_df = df.loc[df["salary"].notna() & (df["status"].astype(str).str.lower() == "placed")].copy()
    if len(salary_df) >= 10:
        Xs = salary_df.drop(columns=["status", "salary"], errors="ignore")
        ys = salary_df["salary"].astype(float)
        nums = Xs.select_dtypes(include=["number"]).columns.tolist()
        cats = [c for c in Xs.columns if c not in nums]
        salary_prep = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), nums),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats)
        ])
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(
            Xs, ys, test_size=0.20, random_state=42
        )
        salary_pipe = Pipeline([("prep", salary_prep), ("model", LinearRegression())])
        salary_pipe.fit(Xs_train, ys_train)
        ys_pred = salary_pipe.predict(Xs_test)
        salary_metrics = {
            "n_salary_records": int(len(salary_df)),
            "mae": float(mean_absolute_error(ys_test, ys_pred)),
            "rmse": float(np.sqrt(mean_squared_error(ys_test, ys_pred))),
            "r2": float(r2_score(ys_test, ys_pred))
        }
        with open(os.path.join(args.outdir, "salary_metrics.json"), "w") as f:
            json.dump(salary_metrics, f, indent=2)
        print("\\n=== LINEAR REGRESSION SALARY RESULTS ===\\n", salary_metrics)
    else:
        print("\\nNot enough observed salary records for a reliable regression split.")

    # Figures
    plt.figure(figsize=(8, 4.8))
    plt.bar(cls["model"], cls["f1"])
    plt.ylabel("F1-score")
    plt.title("Placement classification: held-out test F1")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "classification_f1.png"), dpi=180)
    plt.close()

    counts = df["status"].value_counts()
    plt.figure(figsize=(5.5, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("Number of students")
    plt.title("Placement status distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "placement_distribution.png"), dpi=180)
    plt.close()

    print(f"\\nResults saved to: {os.path.abspath(args.outdir)}")
    print("Use the generated CSV/JSON and figures to replace the clearly marked result placeholders in the journal draft.")

if __name__ == "__main__":
    main()
