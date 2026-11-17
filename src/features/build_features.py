"""
build_features.py  -  Feature Engineering Pipeline
====================================================
MLOps Lesson: "Your feature pipeline must be IDENTICAL at training and inference.
               Training-serving skew silently kills production models."

WHY TRAINING-SERVING SKEW HAPPENS (and how we prevent it)
----------------------------------------------------------
1. Imputation skew:
   Wrong: impute NaN PageValues with the CURRENT batch mean at inference time.
   The imputed value changes every day; the model was never trained on it.
   Correct: save the TRAINING median; use it at every inference call.

2. One-hot encoding skew:
   If "Jan" sessions never appear in inference (Jan is not in the deployment
   window), pd.get_dummies() produces fewer columns than at training time.
   The model receives a vector of the wrong shape and crashes — or worse,
   silently misclassifies all sessions.
   Correct: save the exact column list from training; reindex inference data
   to that list (missing columns → 0, extra columns → dropped).

3. Feature order skew:
   NumPy arrays have no column names. If columns are added/removed between
   training and inference, the model operates on the wrong features entirely.
   Correct: always extract features in the same order, defined by feature_cols.

HOW THIS SOLVES IT
------------------
  * Training medians are saved to `models/feature_pipeline.json`.
    Inference loads them from disk — never recomputes from the inference batch.

  * After OHE, the exact column list is saved. At inference, the DataFrame
    is reindexed to those columns (missing → 0, extra → dropped).

  * training=True populates and saves the pipeline state.
    training=False loads and applies it. One flag, consistent behaviour.

FEATURE DECISIONS
-----------------
Weekend: stored as string "True"/"False" after CSV round-trip.
  Converted to int (1/0) BEFORE one-hot encoding — Weekend is binary,
  so OHE would just create two redundant columns. Direct int is cleaner.

Month: kept as categorical → OHE.
  The model should learn non-linear month effects (May vs November are
  completely different, not adjacent on a linear scale).

VisitorType: OHE with 3 values (Returning_Visitor, New_Visitor, Other).

Numeric features: all treated as continuous floats.
  OperatingSystems, Browser, Region, TrafficType are integer-coded but
  ordinal assumptions are wrong (OS=3 is not "more" than OS=1).
  However, LightGBM handles ordinal codes well because it finds optimal
  split points — no need to OHE these.
"""

import os
import json

import numpy as np
import pandas as pd

from src.config.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, MODEL_DIR,
)

# Path where training-time pipeline metadata is saved
_PIPELINE_META_PATH = os.path.join(MODEL_DIR, "feature_pipeline.json")

# In-memory cache (populated by training, reloaded by inference)
_TRAIN_MEDIANS:      dict = {}
_TRAIN_FEATURE_COLS: list = []


def build_features(df: pd.DataFrame, training: bool = True):
    """
    Transform a raw DataFrame into a numpy feature matrix and label vector.

    Parameters
    ----------
    df       : raw DataFrame as returned by prepare_data()
    training : True  → compute and SAVE medians + column list
               False → LOAD medians + column list from disk (use training values)

    Returns
    -------
    X            : np.ndarray  (n_samples, n_features)  – float32
    y            : np.ndarray  (n_samples,)              – int
    feature_cols : list[str]   – column names for X (for feature importance)
    """
    global _TRAIN_MEDIANS, _TRAIN_FEATURE_COLS

    df = df.copy()

    # ── step 1: load pipeline state for inference ─────────────────────────────
    if not training:
        _load_pipeline_meta()

    # ── step 2: Weekend bool → int ────────────────────────────────────────────
    # WHY INT NOT OHE: Weekend is binary. OHE would create two columns
    # (Weekend_True, Weekend_False) that sum to 1 — redundant.
    # Converting to int keeps it as a single informative feature.
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].map(
            {"True": 1, "False": 0, True: 1, False: 0, "true": 1, "false": 0}
        ).fillna(0).astype(int)

    # ── step 3: median imputation for numeric features ────────────────────────
    # WHY MEDIAN NOT MEAN: Median is robust to outliers.
    # PageValues has a right-skewed distribution with extreme values.
    # The mean imputer would be pulled toward those extremes; median is stable.
    # WHY SAVE: At inference, we impute with the TRAINING median, not the
    # current batch median — the model was trained on the training-time value.
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue

        if training:
            _TRAIN_MEDIANS[col] = float(df[col].median())

        fill_val  = _TRAIN_MEDIANS.get(col, float(df[col].median()))
        n_imputed = int(df[col].isnull().sum())

        if n_imputed > 0:
            df[col] = df[col].fillna(fill_val)
            tag = "computed" if training else "from training set"
            print(f"    [Impute] '{col}': {n_imputed} NaN(s) filled with median "
                  f"{fill_val:.4f} ({tag})")

    # ── step 4: one-hot encode Month and VisitorType ──────────────────────────
    # drop_first=False: keep all dummies so column names are unambiguous.
    # e.g. Month_Nov, Month_Dec, VisitorType_New_Visitor, etc.
    # Weekend is already an int column — exclude from OHE.
    ohe_cols = [c for c in CATEGORICAL_FEATURES if c != "Weekend" and c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # ── step 5: determine feature columns ────────────────────────────────────
    all_cols     = [c for c in df.columns if c != TARGET]
    feature_cols = all_cols

    if training:
        _TRAIN_FEATURE_COLS = feature_cols
        _save_pipeline_meta()
    else:
        # Reindex to EXACTLY the columns seen at training time.
        # Missing columns (a category absent in this batch) → filled with 0.
        # Extra columns (e.g. a new VisitorType value) → silently dropped.
        # This prevents shape mismatches at inference time.
        feature_cols = _TRAIN_FEATURE_COLS
        df = df.reindex(
            columns=feature_cols + ([TARGET] if TARGET in df.columns else []),
            fill_value=0,
        )

    # ── step 6: extract arrays ────────────────────────────────────────────────
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(int) if TARGET in df.columns else np.zeros(len(df), dtype=int)

    return X, y, feature_cols


# ── persistence helpers ───────────────────────────────────────────────────────

def _save_pipeline_meta():
    """Save training medians and feature column list to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "medians":      _TRAIN_MEDIANS,
        "feature_cols": _TRAIN_FEATURE_COLS,
    }
    with open(_PIPELINE_META_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"    [Features] Pipeline metadata saved -> {_PIPELINE_META_PATH}")


def _load_pipeline_meta():
    """Load training medians and feature column list from disk."""
    global _TRAIN_MEDIANS, _TRAIN_FEATURE_COLS
    if _TRAIN_MEDIANS:
        return  # already loaded in this process
    if not os.path.exists(_PIPELINE_META_PATH):
        print(
            "  [WARNING] No feature_pipeline.json found.  "
            "Run the training pipeline first (python main.py)."
        )
        return
    with open(_PIPELINE_META_PATH) as f:
        payload = json.load(f)
    _TRAIN_MEDIANS      = payload["medians"]
    _TRAIN_FEATURE_COLS = payload["feature_cols"]
    print(f"    [Features] Pipeline metadata loaded from {_PIPELINE_META_PATH}")
