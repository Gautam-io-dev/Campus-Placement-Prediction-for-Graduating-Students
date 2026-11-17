CAMPUS-PLACEMENT PREDICTION PROJECT — RESULTS INCLUDED
Student: Gautam Mishra | Roll No.: 24020570012 | Problem No.: 19

Files:
- Gautam_Mishra_Campus_Placement_Journal_Final.pdf: journal-format report with results computed from the uploaded CSV.
- campus_placement_experiments.py: reproducible experiment code.
- results/classification_metrics.csv: held-out classification metrics.
- results/classification_cv_metrics.csv: five-fold stratified cross-validation metrics.
- results/salary_metrics.json: salary regression metrics.
- results/*.png: figures generated from the dataset/results.

Dataset:
Placement_Data_Full_Class.csv (uploaded by the student). For reproduction, keep the CSV in the same folder as the Python script.

Run:
pip install pandas numpy scikit-learn matplotlib
python campus_placement_experiments.py --data Placement_Data_Full_Class.csv

Important interpretation:
The dataset has 215 rows; 148 are placed and 67 are not placed. Salary is missing for non-placed records. Salary regression is conditional on recorded salary among placed candidates. The salary baseline had a negative test R², so it should not be described as a strong predictor. This is a student research draft; verify all citations, disclose AI use according to course policy, and check the institution's formatting requirements before submission.
