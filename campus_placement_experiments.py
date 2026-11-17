# Campus Placement Prediction Project
# Name: Gautam Mishra | Roll No: 24020570012 | Prob 19

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
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

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Placement_Data_Full_Class.csv", help="dataset path")
    parser.add_argument("--outdir", default="results", help="output folder for charts and logs")
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Load data
    df = pd.read_csv(args.data)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'status' not in df.columns or 'salary' not in df.columns:
        print("Error: Required columns missing from CSV.")
        return

    # Drop serial number id column if present
    if 'sl_no' in df.columns:
        df = df.drop(columns=['sl_no'])

    print("\n--- Data Overview ---")
    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nPlacement status counts:\n", df['status'].value_counts())

    # Prep features and target
    # Salary is excluded from classification features since it's only for placed students
    X = df.drop(columns=['status', 'salary'], errors='ignore')
    y = df['status'].astype(str).str.strip().str.lower().map(lambda x: 1 if x == 'placed' else 0)

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocessing pipelines
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB()
    }

    results = []
    trained_models = {}

    print("\n--- Training Classifiers ---")
    for name, clf in models.items():
        if name == "Gaussian Naive Bayes":
            pipe = Pipeline([
                ('prep', preprocessor),
                ('dense', FunctionTransformer(lambda a: a.toarray() if hasattr(a, "toarray") else a)),
                ('model', clf)
            ])
        else:
            pipe = Pipeline([('prep', preprocessor), ('model', clf)])
            
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if proba is not None else float('nan')
        }
        
        results.append(metrics)
        trained_models[name] = pipe
        
        print(f"\nModel: {name}")
        for k, v in metrics.items():
            if k != 'model':
                print(f"  {k}: {v:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Save classification summary
    res_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    res_df.to_csv(os.path.join(args.outdir, "classification_metrics.csv"), index=False)
    print("\nFinal Classification Comparison:\n", res_df)

    # 5-fold CV
    print("\n--- Running Cross-Validation ---")
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for name, pipe in trained_models.items():
        scores = cross_validate(pipe, X, y, cv=cv_splitter, scoring=scoring_metrics, error_score='raise')
        cv_entry = {"model": name}
        for m in scoring_metrics:
            cv_entry[f"{m}_cv_mean"] = float(np.mean(scores[f"test_{m}"]))
        cv_results.append(cv_entry)
        
    cv_summary = pd.DataFrame(cv_results).sort_values('f1_cv_mean', ascending=False)
    cv_summary.to_csv(os.path.join(args.outdir, "classification_cv_metrics.csv"), index=False)
    print(cv_summary)

    # Salary Regression (only for placed students)
    print("\n--- Running Salary Regression ---")
    salary_data = df[(df['status'].astype(str).str.lower() == 'placed') & (df['salary'].notna())].copy()
    
    if len(salary_data) >= 10:
        Xs = salary_data.drop(columns=['status', 'salary'], errors='ignore')
        ys = salary_data['salary'].astype(float)
        
        nums_s = Xs.select_dtypes(include=['number']).columns.tolist()
        cats_s = [c for c in Xs.columns if c not in nums_s]
        
        sal_prep = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), nums_s),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats_s)
        ])
        
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.2, random_state=42)
        
        sal_pipe = Pipeline([('prep', sal_prep), ('model', LinearRegression())])
        sal_pipe.fit(Xs_train, ys_train)
        ys_pred = sal_pipe.predict(Xs_test)
        
        sal_metrics = {
            "n_salary_records": int(len(salary_data)),
            "mae": float(mean_absolute_error(ys_test, ys_pred)),
            "rmse": float(np.sqrt(mean_squared_error(ys_test, ys_pred))),
            "r2": float(r2_score(ys_test, ys_pred))
        }
        
        with open(os.path.join(args.outdir, "salary_metrics.json"), "w") as f:
            json.dump(sal_metrics, f, indent=2)
        print("Salary Regression Results:", sal_metrics)
    else:
        print("Not enough salary records available for regression.")

    # Generate plots
    print("\n--- Saving Plots ---")
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['model'], res_df['f1'], color='skyblue', edgecolor='black')
    plt.ylabel('F1 Score')
    plt.title('Model Comparison - Test F1 Score')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "classification_f1.png"), dpi=150)
    plt.close()

    status_counts = df['status'].value_counts()
    plt.figure(figsize=(5, 4))
    status_counts.plot(kind='bar', color=['lightgreen', 'salmon'], edgecolor='black')
    plt.ylabel('Count')
    plt.title('Placement Status Distribution')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "placement_distribution.png"), dpi=150)
    plt.close()

    print(f"\nAll outputs saved successfully in folder: '{args.outdir}'")

if __name__ == '__main__':
    main()
