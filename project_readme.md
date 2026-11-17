# Campus-Placement Prediction for Graduating Students

**Author:** Gautam Mishra  
**Roll No.:** 24020570012  
**Course:** Machine Learning, B.Sc. (Hons.) Computer Science — Ramanujan College, University of Delhi  
**Allotted Problem No.:** 19  

## Overview
This repository contains the reproducible code and journal-format report evaluating classical machine-learning models on the Campus Recruitment dataset (`Placement_Data_Full_Class.csv`, 215 student records). The project covers two tasks:
1. **Placement Classification:** Predicting binary placement outcome (`Placed` vs. `Not Placed`) using pre-placement academic, demographic, and employability indicators, benchmarked against a `DummyClassifier(strategy="most_frequent")` baseline.
2. **Conditional Salary Regression:** Estimating offered annual salary among the 148 placed candidates with observed salary records, benchmarked against a `DummyRegressor(strategy="mean")` baseline.

## Repository Structure
- `Gautam_Mishra_Campus_Placement_Journal_Revised.pdf` — Final journal-format submission.
- `campus_placement_experiments.py` — End-to-end experiment pipeline (data audit, preprocessing, baselines, 80:20 evaluation, 5-fold stratified cross-validation, salary regression, and figure generation).
- `requirements.txt` — Python package dependencies.
- `Placement_Data_Full_Class.csv` — Input dataset (215 rows, 15 columns).
- `results/` — Generated metrics (`classification_metrics.csv`, `classification_cv_metrics.csv`, `salary_metrics.json`) and plots (`classification_f1.png`, `placement_distribution.png`).

## Quickstart
1. Place `Placement_Data_Full_Class.csv` in the repository root directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the experiment script:
   ```bash
   python campus_placement_experiments.py --data Placement_Data_Full_Class.csv --outdir results
   ```