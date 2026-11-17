"""
loader.py  -  Real Online Shoppers Dataset
==========================================

BUSINESS PROBLEM
----------------
An e-commerce retailer wants to predict which browsing sessions will end
in a purchase, so they can:
  - Serve targeted promotions to high-intent sessions in real time
  - Avoid wasting promotional budget on sessions unlikely to convert
  - Optimise remarketing bids based on predicted purchase probability

Prediction task: binary classification
  1 = session ended in a purchase  (Revenue = True)
  0 = session did not convert      (Revenue = False)

False Negative (miss a buyer):    Lost conversion ≈ $85 average order value
False Positive (flag a non-buyer): Wasted discount/offer ≈ $3
Cost ratio FN:FP ≈ 28:1  → Recall is the primary business lever.

TECHNICAL CHALLENGES
--------------------
1. Class imbalance: 15.5% purchase rate.
   A model that always predicts "no purchase" gets 84.5% accuracy. Useless.
   F1 on the positive class and PR-AUC are the real quality signals.

2. Temporal concept drift: the holiday season (Nov-Dec) is a completely
   different distribution from the rest of the year.
   PageValues spikes, BounceRates drop, SpecialDay is non-zero.
   A model trained on Feb-Oct data degrades in Nov-Dec.

3. Mixed feature types: continuous engagement metrics (BounceRates,
   PageValues), count features (ProductRelated), ordinal/encoded integers
   (Browser, Region, TrafficType), and nominal categories (VisitorType, Month).

4. Zero-inflation: most sessions have zero Informational and Administrative
   page visits. Duration features are right-skewed with a mass at zero.

5. Label delay: in production, the Revenue label for a session is known
   immediately (checkout confirmation). BUT model quality is evaluated on the
   NEXT batch, not the current one — drift detection on input features is
   the early warning system between batch evaluations.

DATASET SOURCE
--------------
Sakar, C.O., Polat, S.O., Katircioglu, M., and Kastro, Y. (2019).
Real-time prediction of online shoppers' purchasing intention using
multilayer perceptron and LSTM recurrent neural networks.
Neural Computing and Applications, 31(10), 6893-6908.

UCI ML Repo: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Dataset ID : 468
License    : CC BY 4.0

TEMPORAL SPLIT LOGIC
--------------------
TRAIN months  : Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct  (~82% of data)
CURRENT months: Nov, Dec  (~18% of data — holiday season)

This split mirrors a real production scenario:
  - Model is trained before the holiday season begins (end of October)
  - It goes live in November, serving real traffic
  - Drift detector monitors the November-December traffic
  - MLOps engineer runs retrain.py to decide whether to update the model
"""

import os
import numpy as np
import pandas as pd

from src.config.config import DATA_DIR, SEED, TRAIN_MONTHS, CURRENT_MONTHS

try:
    from ucimlrepo import fetch_ucirepo
    _UCIMLREPO_AVAILABLE = True
except ImportError:
    _UCIMLREPO_AVAILABLE = False


# ── Download from UCI ─────────────────────────────────────────────────────────

def _download_from_ucimlrepo() -> pd.DataFrame:
    """Download UCI Online Shoppers Purchasing Intention (id=468)."""
    print("  Fetching UCI ML Repo dataset 468 (Online Shoppers Purchasing Intention)...")
    dataset = fetch_ucirepo(id=468)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise dtypes and target encoding from the raw UCI download."""
    df = df.copy()

    # Revenue may be bool True/False from UCI — convert to int
    if "Revenue" in df.columns:
        df["Revenue"] = df["Revenue"].astype(int)

    # Weekend may be bool — convert to str for consistent OHE treatment
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(str)

    return df


# ── Drift simulation for current batch ───────────────────────────────────────

def _apply_holiday_shift(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """
    Amplify the holiday-season distribution shift in the current batch.

    In the real dataset, Nov-Dec already differ from Feb-Oct.
    This function is used ONLY when working with the synthetic fallback,
    where the shift must be injected manually. For real UCI data, the
    temporal split itself provides the natural drift.

    What shifts during the holiday season (Nov-Dec):
    ------------------------------------------------
    * PageValues:  UP   (+60-80%) — users arrive with more purchase intent
    * BounceRates: DOWN (-20-30%) — engaged holiday shoppers rarely bounce
    * ExitRates:   DOWN (-20-30%) — same engagement effect
    * ProductRelated: UP (+20-30%) — more product page browsing
    * SpecialDay: UP   (more non-zero values — promotional days)
    * New_Visitor fraction: UP (gift buyers, first-time visitors)

    What does NOT shift:
    --------------------
    * OperatingSystems, Browser, Region — technology/geography stable
    * TrafficType distribution — marketing channel mix stable month-to-month
    """
    rng = np.random.default_rng(seed)
    df  = df.copy()
    n   = len(df)

    # PageValues spikes: 60% of sessions have boosted values
    mask = rng.random(n) < 0.60
    df.loc[mask, "PageValues"] = df.loc[mask, "PageValues"] * rng.uniform(1.5, 2.2, mask.sum())

    # BounceRates and ExitRates decrease
    df["BounceRates"] = df["BounceRates"] * np.clip(rng.normal(0.78, 0.08, n), 0.5, 1.0)
    df["ExitRates"]   = df["ExitRates"]   * np.clip(rng.normal(0.80, 0.07, n), 0.5, 1.0)

    # ProductRelated browsing increases
    df["ProductRelated"]          = df["ProductRelated"]          * rng.uniform(1.15, 1.35, n)
    df["ProductRelated_Duration"] = df["ProductRelated_Duration"] * rng.uniform(1.10, 1.30, n)

    # SpecialDay: more sessions are near a promotional day
    extra_special = rng.random(n) < 0.25
    df.loc[extra_special, "SpecialDay"] = rng.choice(
        [0.2, 0.4, 0.6, 0.8, 1.0], extra_special.sum()
    )

    # More New_Visitors (gift shoppers)
    flip_to_new = rng.random(n) < 0.08
    df.loc[flip_to_new, "VisitorType"] = "New_Visitor"

    # Month label: holiday
    df["Month"] = rng.choice(["Nov", "Dec"], n, p=[0.55, 0.45])

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_shopper_data(seed: int = SEED) -> tuple:
    """
    Load or generate the Online Shoppers Purchasing Intention dataset.

    Priority:
      1. Load data/raw.csv if it exists (cached prior download)
      2. Try ucimlrepo.fetch_ucirepo(id=468) — real 12,330-row dataset
      3. Fall back to calibrated synthetic generation (src/data/generate.py)

    Returns
    -------
    (train_df, current_df): DataFrames with identical schema
      train_df   : sessions from TRAIN_MONTHS (Feb-Oct)
      current_df : sessions from CURRENT_MONTHS (Nov-Dec)
    """
    from src.data.generate import generate_shopper_data

    raw_path = os.path.join(DATA_DIR, "raw.csv")
    df_raw   = None

    # ── attempt real data ─────────────────────────────────────────────────────
    if os.path.exists(raw_path):
        print(f"  Loading cached dataset from {raw_path}")
        df_raw = pd.read_csv(raw_path)

    elif _UCIMLREPO_AVAILABLE:
        try:
            df_raw = _download_from_ucimlrepo()
            df_raw = _clean(df_raw)
            os.makedirs(DATA_DIR, exist_ok=True)
            df_raw.to_csv(raw_path, index=False)
            print(f"  Real data downloaded and cached → {raw_path}")
        except Exception as exc:
            print(f"  [WARN] UCI download failed: {exc}")
            print("         Falling back to calibrated synthetic data.")
            df_raw = None
    else:
        print("  [INFO] ucimlrepo not installed. Using calibrated synthetic fallback.")
        print("         Install with: pip install ucimlrepo")

    # ── split by month ────────────────────────────────────────────────────────
    if df_raw is not None and len(df_raw) > 0:
        df_raw = _clean(df_raw)

        train_df   = df_raw[df_raw["Month"].isin(TRAIN_MONTHS)].reset_index(drop=True)
        current_df = df_raw[df_raw["Month"].isin(CURRENT_MONTHS)].reset_index(drop=True)

        print(
            f"  [REAL DATA] UCI Online Shoppers — "
            f"train (Feb-Oct)={len(train_df):,}, current (Nov-Dec)={len(current_df):,}"
        )
    else:
        # Synthetic fallback: generate non-holiday train + holiday-shifted current
        train_df   = generate_shopper_data(n=10000, holiday=False, seed=seed)
        current_df = generate_shopper_data(n=2400,  holiday=True,  seed=seed + 1)
        print(
            f"  [SYNTHETIC] Calibrated to UCI profile — "
            f"train={len(train_df):,}, current={len(current_df):,}"
        )

    pos_train   = train_df["Revenue"].mean()
    pos_current = current_df["Revenue"].mean()
    print(f"  Purchase rate — train: {pos_train:.1%}  current (holiday): {pos_current:.1%}")
    if pos_current > pos_train + 0.03:
        print(
            "  [NOTE] Holiday batch has higher purchase rate — expected. "
            "Drift detection will flag feature distribution shifts."
        )

    return train_df, current_df
