"""
evaluate.py  -  Model Evaluation & Promotion Gates
====================================================
MLOps Lesson: "Accuracy alone is not a deployment criterion.
               Evaluating on TRAINING data is not evaluation at all."

THE TWO FATAL EVALUATION MISTAKES
----------------------------------
Mistake 1 — Accuracy trap:
  15.5% positive rate. A model that always predicts 'no purchase' gets
  84.5% accuracy. It has learned NOTHING. F1 on the minority class and
  PR-AUC are the only metrics that prove the model actually identifies buyers.

Mistake 2 — Train-set evaluation (the bug this file fixes):
  A RandomForest with max_depth=None memorises the training data perfectly.
  Evaluating on the same data it was trained on will always show F1=1.000.
  This is NOT model performance — it is memorisation.

  CORRECT APPROACH: The pipeline splits the training months (Feb-Oct) into
  an 80% fit set and a 20% hold-out test set. Evaluation is ALWAYS on the
  hold-out test set — data the model has NEVER seen.

  pipeline: train_df (Feb-Oct)
            └─ 80% → model.fit(X_train, y_train)
            └─ 20% → evaluate(model, X_test, y_test)   ← this file

SLICE EVALUATION
----------------
A model can look fine on the full test set but perform badly on a subgroup.
Example: F1=0.75 overall, but F1=0.45 for New_Visitor sessions.
Deploying this model harms new visitors — they never get targeted offers
even when they have high purchase intent.

Slice evaluation catches this before deployment. It is the foundation of
fairness-aware ML — you can't fix what you don't measure.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.config.config import (
    F1_THRESHOLD,
    RECALL_THRESHOLD,
    PRECISION_THRESHOLD,
    AUC_ROC_THRESHOLD,
    AUC_PR_THRESHOLD,
)


class GateFailure(Exception):
    """Raised when the model fails one or more promotion gates."""


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list = None,
    label: str = "Test set",
) -> dict:
    """
    Compute metrics on a HELD-OUT set and print a formatted report.

    This function must ONLY be called with TEST data — never training data.
    Call it with the 20% hold-out split, not the data used in model.fit().

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc_roc, auc_pr
    """
    preds = model.predict(X)

    # Probability scores for AUC metrics (requires predict_proba)
    try:
        proba = model.predict_proba(X)[:, 1]
        auc_roc = round(float(roc_auc_score(y, proba)), 4)
        auc_pr  = round(float(average_precision_score(y, proba)), 4)
    except Exception:
        auc_roc = None
        auc_pr  = None

    metrics = {
        "accuracy":  round(float(accuracy_score(y, preds)),                        4),
        "precision": round(float(precision_score(y, preds, zero_division=0)),      4),
        "recall":    round(float(recall_score(y, preds, zero_division=0)),          4),
        "f1":        round(float(f1_score(y, preds, zero_division=0)),              4),
        "auc_roc":   auc_roc,
        "auc_pr":    auc_pr,
    }

    pos_rate = float(y.mean())

    print(f"\n  Evaluation on: {label}  "
          f"(n={len(y):,}, positive rate={pos_rate:.1%})")
    print(f"\n  {'Metric':<20} {'Value':>7}   {'Gate':>14}  {'Status':>8}")
    print(f"  {'-'*20}  {'-'*7}   {'-'*14}  {'-'*8}")

    def _gate_str(val, threshold, higher_is_better=True):
        if val is None:
            return "N/A", "  n/a"
        op = ">=" if higher_is_better else "<="
        passed = val >= threshold if higher_is_better else val <= threshold
        return f"{op} {threshold}", "PASS" if passed else "FAIL"

    rows = [
        ("Accuracy",        metrics["accuracy"],  None,                 "informational"),
        ("F1 (minority)",   metrics["f1"],        F1_THRESHOLD,         "gate"),
        ("Precision",       metrics["precision"], PRECISION_THRESHOLD,  "gate"),
        ("Recall",          metrics["recall"],    RECALL_THRESHOLD,     "gate"),
        ("AUC-ROC",         metrics["auc_roc"],   AUC_ROC_THRESHOLD,    "gate"),
        ("AUC-PR",          metrics["auc_pr"],    AUC_PR_THRESHOLD,     "gate"),
    ]

    for name, val, threshold, kind in rows:
        if val is None:
            print(f"  {name:<20} {'N/A':>7}   {'':>14}  {'n/a':>8}")
            continue
        if kind == "informational":
            print(f"  {name:<20} {val:>7.4f}   {'(informational)':>14}")
        else:
            gate_str, status = _gate_str(val, threshold)
            print(f"  {name:<20} {val:>7.4f}   {gate_str:>14}  {status:>8}")

    # WHY WE SHOW BOTH F1 AND AUC-ROC
    # --------------------------------
    # AUC-ROC measures ranking quality across all thresholds — good for
    # understanding model power. F1 measures performance AT the deployment
    # threshold (0.5) — good for understanding real-world impact.
    # A model can have great AUC but poor F1 at threshold=0.5 (poorly calibrated).
    # Both matter; F1 is the business gate.

    if feature_cols is not None and hasattr(model, "feature_importances_"):
        _print_feature_importance(model, feature_cols)

    return metrics


def _print_feature_importance(model, feature_cols: list):
    """Print the top-10 features by LightGBM split gain."""
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(10)

    print("\n  Top-10 feature importances (LightGBM split gain):")
    total = top.sum()
    for feat, imp in top.items():
        pct = imp / (total + 1e-9)
        bar = "|" * max(1, int(pct * 50))
        print(f"    {feat:<40} {pct:>5.1%}  {bar}")

    print()
    print("  Interpretation:")
    top_name = top.index[0]
    print(f"    '{top_name}' is the dominant predictor.")
    if "PageValues" in top.index[:3]:
        print("    PageValues near top: users who viewed high-value product pages "
              "are very likely to buy — expected from the business logic.")
    if "BounceRates" in top.index[:5] or "ExitRates" in top.index[:5]:
        print("    BounceRates/ExitRates near top: engagement quality matters more "
              "than raw visit counts.")


# ── promotion gate ────────────────────────────────────────────────────────────

def gate(metrics: dict) -> bool:
    """
    Block model promotion if any threshold is not met.

    WHY A HARD GATE (not just a warning)
    --------------------------------------
    In a CI/CD pipeline, warnings get ignored. A hard gate turns a bad model
    into a pipeline failure that cannot be silently deployed. The engineer
    must explicitly investigate and either fix the model or change the threshold
    with a documented business justification.

    A gate failure on F1 means the model cannot reliably identify buyers.
    Deploying it means wasted promotional spend and missed conversions.

    Returns True if all gates pass.
    Raises GateFailure if any gate fails.
    """
    failures = []

    checks = [
        ("F1 (minority)",   metrics.get("f1"),      F1_THRESHOLD,        ">="),
        ("Recall",          metrics.get("recall"),   RECALL_THRESHOLD,    ">="),
        ("Precision",       metrics.get("precision"),PRECISION_THRESHOLD, ">="),
        ("AUC-ROC",         metrics.get("auc_roc"),  AUC_ROC_THRESHOLD,   ">="),
        ("AUC-PR",          metrics.get("auc_pr"),   AUC_PR_THRESHOLD,    ">="),
    ]

    for name, val, threshold, op in checks:
        if val is None:
            continue
        passed = val >= threshold if op == ">=" else val <= threshold
        if not passed:
            failures.append(f"  {name}: {val:.4f} {op} {threshold} FAILED")

    if failures:
        print("\n  [GATE FAIL] Model blocked from promotion:")
        for f in failures:
            print(f"    [x] {f}")
        raise GateFailure(
            "Model did not meet all promotion thresholds. "
            "Consider: (1) checking for data leakage, (2) adjusting "
            "scale_pos_weight, (3) using retrain.py --strategy reexperiment "
            "to search for better hyperparameters."
        )

    print("\n  [GATE PASS] All thresholds met. Model approved for staging.")
    return True


# ── slice evaluation ──────────────────────────────────────────────────────────

def slice_evaluation(
    model,
    X:            np.ndarray,
    y:            np.ndarray,
    df_raw:       pd.DataFrame,
    slice_col:    str,
    feature_cols: list,
) -> dict:
    """
    Evaluate model performance separately for each value of `slice_col`.

    This answers: "Is the model equally good for ALL visitor subgroups?"

    For example, if F1 is 0.76 for Returning_Visitor but only 0.52 for
    New_Visitor, the model under-serves new visitors — they never get
    targeted promotions even when they intend to buy.

    Parameters
    ----------
    df_raw    : original DataFrame before feature engineering (used for slicing)
    slice_col : column to slice on (e.g. 'VisitorType', 'Month')

    Returns
    -------
    dict mapping group name -> metrics dict
    """
    if slice_col not in df_raw.columns:
        print(f"  [Slice] Column '{slice_col}' not found — skipping.")
        return {}

    preds  = model.predict(X)
    groups = sorted(df_raw[slice_col].dropna().astype(str).unique())

    print(f"\n  Slice by '{slice_col}':")
    print(f"  {'Group':<22} {'N':>5}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'AUC-ROC':>8}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

    results = {}
    try:
        proba = model.predict_proba(X)[:, 1]
        has_proba = True
    except Exception:
        has_proba = False

    for group in groups:
        mask = (df_raw[slice_col].astype(str) == group).values
        if mask.sum() < 20:
            continue

        y_g = y[mask]
        p_g = preds[mask]

        f1   = f1_score(y_g, p_g, zero_division=0)
        prec = precision_score(y_g, p_g, zero_division=0)
        rec  = recall_score(y_g, p_g, zero_division=0)

        if has_proba and len(np.unique(y_g)) > 1:
            auc_str = f"{roc_auc_score(y_g, proba[mask]):.3f}"
        else:
            auc_str = "  n/a"

        flag = "  << low F1" if f1 < F1_THRESHOLD - 0.05 else ""
        print(
            f"  {group:<22} {mask.sum():>5}  {f1:>6.3f}  {prec:>6.3f}  "
            f"{rec:>6.3f}  {auc_str:>8}{flag}"
        )

        results[group] = {
            "f1": f1, "precision": prec, "recall": rec, "n": int(mask.sum())
        }

    return results
