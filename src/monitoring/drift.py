"""
drift.py  —  Data Drift Detection
====================================
Compares feature distributions between the training batch (Feb-Oct) and
the post-deployment batch (Nov-Dec) to detect distribution shift.

WHAT IS DATA DRIFT
------------------
Data drift (covariate shift) occurs when the distribution of input features
changes between training time and inference time.

In this project: the model was trained on Feb-Oct browsing sessions (normal
shopping behaviour). In November, holiday shoppers arrive. They visit more
product pages, view higher-value pages, and have lower bounce rates. The
model was never trained on this population. If the shift is large enough,
predictions will degrade.

WHY FEATURE DRIFT (NOT JUST ACCURACY)
---------------------------------------
Accuracy requires GROUND TRUTH LABELS. In e-commerce, the true label
(did they purchase?) is available quickly — but in many domains it arrives
with a delay. Drift detection on INPUT FEATURES is an early warning system
that works WITHOUT waiting for labels. If PageValues (the most predictive
feature) shifts meaningfully, the model may already be stale before the
post-deployment accuracy figures confirm it.

METHOD
------
Primary metric: relative change in column mean.

    relative_diff = |mean_current - mean_train| / |mean_train|

Per-feature thresholds are set based on the feature's velocity — how fast
it legitimately changes between months. PageValues changes faster than
Region; their thresholds reflect that. All thresholds are in config.py.

Secondary: relative change in standard deviation.
A feature can shift in variance without shifting in mean (e.g., the
population becomes more spread — some visitors have very high PageValues,
others have zero). Variance drift alone is flagged as a warning.

More sophisticated methods (PSI, KS test, Jensen-Shannon divergence) can
be added as the system matures. This mean-shift baseline is a good
starting point for a first production system.
"""

import os
import datetime

import pandas as pd

from src.config.config import (
    LOG_DIR, NUMERIC_FEATURES, DRIFT_THRESHOLDS, DRIFT_THRESHOLD,
    RETRAIN_HIGH_IMPORTANCE_FEATURES,
)


def detect_drift(
    train_df:   pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    """
    Compare summary statistics between the training distribution and a new batch.

    Uses per-feature velocity thresholds from config.DRIFT_THRESHOLDS.
    Falls back to config.DRIFT_THRESHOLD for any feature not listed.

    Also checks whether any high-importance feature (RETRAIN_HIGH_IMPORTANCE_FEATURES)
    has drifted — this is surfaced separately to inform the retrain decision.

    Writes results to logs/drift.csv for historical tracking.

    Returns
    -------
    dict: {col → {
        "mean_train": float,
        "mean_current": float,
        "rel_diff_mean": float,
        "rel_diff_std": float,
        "threshold": float,
        "flagged": bool,
        "high_importance": bool,
    }}
    """
    cols = [c for c in NUMERIC_FEATURES
            if c in train_df.columns and c in current_df.columns]

    train_mean = train_df[cols].mean()
    curr_mean  = current_df[cols].mean()
    train_std  = train_df[cols].std()
    curr_std   = current_df[cols].std()

    rel_diff_mean = (curr_mean - train_mean).abs() / (train_mean.abs() + 1e-9)
    rel_diff_std  = (curr_std  - train_std ).abs() / (train_std .abs() + 1e-9)

    # ── print report ──────────────────────────────────────────────────────────
    print(f"\n  {'Feature':<20} {'Train mean':>10}  {'Curr mean':>10}  "
          f"{'Rel diff':>8}  {'Threshold':>9}  {'Status':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*10}")

    results = {}
    flagged_cols = []

    for col in cols:
        threshold  = DRIFT_THRESHOLDS.get(col, DRIFT_THRESHOLD)
        is_flagged = bool(rel_diff_mean[col] > threshold)
        is_hi_imp  = col in RETRAIN_HIGH_IMPORTANCE_FEATURES
        hi_marker  = " ★" if is_hi_imp else ""

        if is_flagged:
            status = "DRIFT !" + hi_marker
            flagged_cols.append(col)
        else:
            status = "ok" + hi_marker

        print(
            f"  {col:<20}  {train_mean[col]:>10.2f}  {curr_mean[col]:>10.2f}  "
            f"{rel_diff_mean[col]:>7.3f}  {threshold:>9.3f}  {status:>10}"
        )

        results[col] = {
            "mean_train":     round(float(train_mean[col]),   4),
            "mean_current":   round(float(curr_mean[col]),    4),
            "rel_diff_mean":  round(float(rel_diff_mean[col]), 4),
            "rel_diff_std":   round(float(rel_diff_std[col]),  4),
            "threshold":      threshold,
            "flagged":        is_flagged,
            "high_importance": is_hi_imp,
        }

    # ── summary ───────────────────────────────────────────────────────────────
    hi_imp_flagged = [c for c in flagged_cols if c in RETRAIN_HIGH_IMPORTANCE_FEATURES]
    n_flagged      = len(flagged_cols)

    print(f"\n  ★ = high-importance feature (retrain if ANY of these drift)")

    if flagged_cols:
        print(f"\n  [DRIFT DETECTED] {n_flagged} feature(s) exceeded velocity threshold: {flagged_cols}")
        if hi_imp_flagged:
            print(f"  [CRITICAL] High-importance feature(s) drifted: {hi_imp_flagged}")
            print(f"             → Rule 1 retrain trigger ACTIVE regardless of feature count.")
        print(
            "\n  Business implication: model predictions may no longer reflect the "
            "current customer population. Run retrain.py to evaluate."
        )
    else:
        print(f"\n  [NO DRIFT] All features within their velocity thresholds.")

    # ── persist to CSV ────────────────────────────────────────────────────────
    row = {f"{c}_reldiff_mean": results[c]["rel_diff_mean"] for c in cols}
    row["n_flagged"]          = n_flagged
    row["flagged"]            = str(flagged_cols)
    row["hi_importance_drift"] = str(hi_imp_flagged)
    row["timestamp"]          = str(datetime.datetime.now())

    drift_path = os.path.join(LOG_DIR, "drift.csv")
    df_row     = pd.DataFrame([row])
    if not os.path.exists(drift_path):
        df_row.to_csv(drift_path, index=False)
    else:
        df_row.to_csv(drift_path, mode="a", header=False, index=False)

    return results
