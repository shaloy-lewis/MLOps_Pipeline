# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
baseline.py — Common Pitfalls in ML Without MLOps Discipline
=============================================================
Demonstrates the issues that arise when building a classifier without
proper MLOps practices. Every block below runs without throwing an error.
Every block produces output that looks reasonable. That is precisely the problem.

    python baseline.py

After reviewing the output, run:
    python model_selection.py   # rigorous model comparison with correct metric
    python main.py              # production pipeline with all checks in place
"""

import warnings
warnings.filterwarnings("ignore")
# sklearn convergence/feature name warnings are part of the expected baseline output
# but we suppress them here to keep the printed results readable
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
for mod in ["sklearn", "lightgbm", "xgboost"]:
    warnings.filterwarnings("ignore", module=mod)

import numpy as np
import pandas as pd
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics        import (
    accuracy_score, classification_report,
    f1_score, recall_score, confusion_matrix,
)

print("=" * 70)
print("  E-Commerce Purchase Prediction — Baseline Analysis")
print("  UCI Online Shoppers Purchasing Intention Dataset")
print("=" * 70)
print("""
  CONTEXT
  -------
  It is October. An e-commerce retailer wants to predict which browsing
  sessions will end in a purchase, so the platform can serve targeted
  offers to high-intent visitors before they leave.

  The data science team has 7,317 sessions collected from February to
  October — everything the company has recorded so far.

  A junior data scientist takes a first pass at the problem.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Pitfall 1: Hardcoded paths with no error handling
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data from hardcoded relative path...")
try:
    df = pd.read_csv("data/train.csv")
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")
    print(f"  Sessions span February to October — this is all the data that exists.")
    print(f"  Target column: 'Revenue'  |  Type: {df['Revenue'].dtype}")
except FileNotFoundError:
    print("  data/train.csv not found. Run main.py first to generate the dataset.")
    sys.exit(0)

df["Revenue"] = df["Revenue"].astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Pitfall 2: Naive feature engineering
#
# A first attempt at preprocessing:
#   - Drop Administrative and Informational pages ("not about products, skip")
#   - Label-encode Month and VisitorType — treats nominal categories as numbers,
#     implying that February < March < ... < December as a continuous scale,
#     and that 'Returning_Visitor' < 'New_Visitor' as an ordered relationship.
#     Both assumptions are incorrect. This silently introduces wrong signal.
#   - No missing value handling — just let the model deal with it
#   - Nothing is saved — medians, encoders, column order all discarded
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Feature engineering (naive approach)...")

# Drop columns that seem less important without analysis
df_feat = df.drop(columns=["Administrative", "Administrative_Duration",
                             "Informational", "Informational_Duration"])
print(f"  Dropped 4 columns ('not important enough'). Remaining: {len(df_feat.columns)}")

# Label-encode categorical columns
# This assigns integers to categories: Feb=0, Mar=1, Oct=2, ...
# The model will treat these as ordered numbers, which is wrong for months.
le_month   = LabelEncoder()
le_visitor = LabelEncoder()
le_weekend = LabelEncoder()

df_feat = df_feat.copy()
df_feat["Month"]       = le_month.fit_transform(df_feat["Month"].astype(str))
df_feat["VisitorType"] = le_visitor.fit_transform(df_feat["VisitorType"].astype(str))
df_feat["Weekend"]     = le_weekend.fit_transform(df_feat["Weekend"].astype(str))

print(f"  Month encoding    : {dict(zip(le_month.classes_, le_month.transform(le_month.classes_)))}")
print(f"  VisitorType encoding: {dict(zip(le_visitor.classes_, le_visitor.transform(le_visitor.classes_)))}")
print(f"  These are nominal categories encoded as integers.")
print(f"  The model will treat Feb(0) < Mar(1) < ... as an ordinal scale.")
print(f"  That ordering doesn't exist — this introduces incorrect signal.")
print(f"\n  Feature pipeline (encoders, medians) not saved anywhere.")
print(f"  At inference time, the encoder must be reconstructed from scratch.")

feature_cols = [c for c in df_feat.columns if c != "Revenue"]
X = df_feat[feature_cols].values
y = df_feat["Revenue"].values


# ─────────────────────────────────────────────────────────────────────────────
# Pitfall 3: Quick model sweep — right tool, wrong metric
#
# A rapid cross-validated comparison across several classifiers is a
# sensible starting point. The problem is sorting by Accuracy on a
# dataset with 12% positive rate. A model that predicts "no purchase"
# for every session achieves 87.7% accuracy. The leaderboard looks
# impressive. The models are catching almost no actual buyers.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Rapid model sweep — sorted by Accuracy (wrong metric for this problem)...")
print("  Tools like PyCaret automate this kind of comparison.")
print("  The default sort metric is Accuracy. On imbalanced data, this misleads.\n")

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.model_selection import cross_val_predict, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

sweep_models = [
    ("Logistic Regression",  LogisticRegression(max_iter=500)),
    ("Decision Tree",        DecisionTreeClassifier(max_depth=8)),
    ("Naive Bayes",          GaussianNB()),
    ("Random Forest",        RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)),
]

sweep_results = []
for name, clf in sweep_models:
    preds_cv = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    acc = (preds_cv == y).mean()
    f1  = f1_score(y, preds_cv, zero_division=0)
    rec = recall_score(y, preds_cv, zero_division=0)
    sweep_results.append({"model": name, "accuracy": acc, "f1_buyers": f1, "recall_buyers": rec})

results_df = pd.DataFrame(sweep_results)

print("  Leaderboard sorted by ACCURACY:")
print(f"  {'Model':<24} {'Accuracy':>10}  {'F1 (buyers)':>12}  {'Recall (buyers)':>16}")
print(f"  {'-'*24}  {'-'*10}  {'-'*12}  {'-'*16}")
for _, row in results_df.sort_values("accuracy", ascending=False).iterrows():
    print(f"  {row['model']:<24} {row['accuracy']:>10.4f}  {row['f1_buyers']:>12.4f}  {row['recall_buyers']:>16.4f}")

print()
print("  Re-sorted by F1 on buyers:")
print(f"  {'Model':<24} {'Accuracy':>10}  {'F1 (buyers)':>12}  {'Recall (buyers)':>16}")
print(f"  {'-'*24}  {'-'*10}  {'-'*12}  {'-'*16}")
for _, row in results_df.sort_values("f1_buyers", ascending=False).iterrows():
    print(f"  {row['model']:<24} {row['accuracy']:>10.4f}  {row['f1_buyers']:>12.4f}  {row['recall_buyers']:>16.4f}")

best_by_acc = results_df.sort_values("accuracy", ascending=False).iloc[0]
best_by_f1  = results_df.sort_values("f1_buyers", ascending=False).iloc[0]
print(f"\n  #1 by Accuracy : {best_by_acc['model']} ({best_by_acc['accuracy']:.1%} acc,  "
      f"{best_by_acc['f1_buyers']:.2f} F1 on buyers)")
print(f"  #1 by F1       : {best_by_f1['model']} ({best_by_f1['accuracy']:.1%} acc,  "
      f"{best_by_f1['f1_buyers']:.2f} F1 on buyers)")
print(f"\n  Accuracy makes the naive model look competitive.")
print(f"  F1 on buyers tells a very different story.")


# ─────────────────────────────────────────────────────────────────────────────
# Pitfall 4: Picking the accuracy winner and reporting it to stakeholders
#
# Random Forest is selected (highest accuracy). It is trained on 80% of the
# data and evaluated on 20%. No stratification, no class imbalance handling.
# The result: ~88% accuracy. This gets presented to the business as success.
# Nobody looks at the confusion matrix or the per-class F1.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Training the 'winner' from the sweep on a proper train/test split...")
print("  Model selected: Random Forest (highest accuracy in sweep)")
print("  No stratify=y, no class_weight. 80/20 split.\n")

# No random_state: results vary between runs (no reproducibility)
# No stratify: the minority class may be unevenly split
# No class_weight: RF will strongly favour the majority class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
preds_test = model.predict(X_test)

acc_test = accuracy_score(y_test, preds_test)
f1_test  = f1_score(y_test, preds_test, zero_division=0)
rec_test = recall_score(y_test, preds_test, zero_division=0)

print(f"  Test set size : {len(y_test):,} rows")
print(f"  Positive rate in test set : {y_test.mean():.1%}  "
      f"(no stratify — may not match training distribution)")
print()
print(f"  What gets reported to stakeholders:")
print(f"    Accuracy : {acc_test:.1%}  ← presented as the key result")
print()
print(f"  What doesn't get mentioned:")
print(f"    F1 on buyers  : {f1_test:.4f}")
print(f"    Recall on buyers: {rec_test:.4f}  "
      f"({'only ' + str(int(rec_test * 100)) + '% of actual buyers identified' if rec_test < 0.6 else 'acceptable'})")

cm = confusion_matrix(y_test, preds_test)
tn, fp, fn, tp = cm.ravel()
print()
print(f"  Confusion matrix:")
print(f"    True Negatives  (correctly said 'no buy') : {tn:,}")
print(f"    False Positives (incorrectly flagged)      : {fp:,}")
print(f"    False Negatives (buyers we MISSED)         : {fn:,}  ← business cost: {fn} × $85 = ${fn*85:,}")
print(f"    True Positives  (buyers correctly found)   : {tp:,}")
print()
print(f"  The model missed {fn:,} buyers in this test batch alone.")
print(f"  Each missed buyer represents ~$85 in lost revenue.")
print(f"  88% accuracy is the headline. The confusion matrix is the reality.")

print()
print(f"  Classification report:")
report = classification_report(y_test, preds_test, target_names=["No Purchase", "Purchase"])
for line in report.split("\n"):
    print(f"    {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Pitfall 5: No feature pipeline saved
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] No feature pipeline saved (training-serving skew)...")
print("  LabelEncoders and dropped columns are not saved anywhere.")
print("  At inference time, the engineer reconstructs the pipeline from memory.")
print("  In November, the inference batch has new Month values ('Nov', 'Dec').")
print("  The LabelEncoder fitted on Feb-Oct data has never seen 'Nov' or 'Dec'.")
print("  It will raise an error — or worse, silently encode them as 0.")
print("  The model receives the wrong inputs. Predictions are quietly wrong.")


# ─────────────────────────────────────────────────────────────────────────────
# Pitfall 6: No experiment tracking
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] No experiment tracking...")
print("  This run used n_estimators=100, depth=unlimited, no class_weight.")
print("  None of that is recorded. In 3 weeks when someone asks 'which config")
print("  was deployed?' the answer is: check the commit, hope it's there.")
print("  There is no experiments.csv, no run ID, no parameter–metric link.")


# ─────────────────────────────────────────────────────────────────────────────
# Pitfall 7: No drift monitoring
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] No drift monitoring...")
print("  Model trained on Feb-Oct (normal shopping). Goes live in November.")
print("  The post-deployment Nov-Dec traffic looks like this vs training:")

try:
    current_df = pd.read_csv("data/current.csv")
    drift_feats = ["PageValues", "ProductRelated", "BounceRates", "ExitRates"]
    print(f"\n  {'Feature':<20} {'Train mean':>12}  {'Nov-Dec mean':>12}  {'Shift':>8}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*8}")
    for feat in drift_feats:
        if feat in df.columns and feat in current_df.columns:
            train_m = df[feat].mean()
            curr_m  = current_df[feat].mean()
            shift   = (curr_m - train_m) / (abs(train_m) + 1e-9)
            print(f"  {feat:<20} {train_m:>12.3f}  {curr_m:>12.3f}  {shift:>+7.0%}")
    print()
except FileNotFoundError:
    print("  (Run main.py first to generate data/current.csv)")
    print("  Holiday shift: PageValues +34%, ProductRelated +48%,")
    print("                 BounceRates +16%, ExitRates +12%")

print("  These are the top-4 features by model importance.")
print("  No monitoring detects this. Performance degrades silently.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
pos_rate = df["Revenue"].mean()
naive_acc = 1 - pos_rate

print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)

pitfalls = [
    ("1", "Hardcoded paths",                "Breaks on any machine except the author's"),
    ("2", "Wrong feature engineering",      "LabelEncoder on nominal = false ordinal signal"),
    ("3", "Wrong metric in rapid sweep",    f"Accuracy #1 model catches only {int(best_by_acc['recall_buyers']*100)}% of buyers"),
    ("4", "Model selected by accuracy",     f"{acc_test:.1%} accuracy reported, {fn} buyers missed (~${fn*85:,} cost)"),
    ("5", "No feature pipeline saved",      "LabelEncoder fails on unseen Nov-Dec months"),
    ("6", "No experiment tracking",         "No record of config, no reproducibility"),
    ("7", "No drift monitoring",            "Model silently fails in the holiday season"),
]

print(f"\n  {'#':<3} {'Pitfall':<32} {'Consequence'}")
print(f"  {'-'*3} {'-'*32} {'-'*38}")
for num, pitfall, consequence in pitfalls:
    print(f"  {num:<3} {pitfall:<32} {consequence}")

print()
print(f"  None of these issues threw an error.")
print(f"  The accuracy number ({acc_test:.1%}) is real. It looks reasonable.")
print(f"  The business implication ({fn} missed buyers, ~${fn*85:,} lost) is invisible.")
print(f"  This is the difference between code that runs and code that works.")
print()
print("  Next steps:")
print("    python model_selection.py   # rigorous comparison with correct metric")
print("    python main.py              # production pipeline that addresses all issues")
print("=" * 70)
