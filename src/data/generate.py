"""
generate.py  -  Synthetic Online Shoppers Dataset
==================================================
Generates a calibrated synthetic dataset matching the UCI Online Shoppers
Purchasing Intention profile (Sakar et al. 2019).

Used as a FALLBACK when ucimlrepo is unavailable or the download fails.
All distributions are calibrated to match the UCI dataset's Table 1
summary statistics and known feature importances.

WHY THIS DATASET TEACHES EVERY MLOps STAGE
-------------------------------------------
  * Class imbalance (15.5% positive) → teaches why accuracy is useless,
    why F1/PR-AUC are the right metrics, and why class_weight matters.

  * Zero-inflated features (most sessions have 0 Informational pages)
    → teaches imputation strategy and feature distribution analysis.

  * PageValues is the dominant predictor (users who view product pages
    with non-zero value almost always buy) → teaches feature importance.

  * holiday=True distribution shift:
    - PageValues spikes (higher purchase intent)
    - BounceRates and ExitRates drop (more engaged visitors)
    - SpecialDay is non-zero more often (promotional calendar)
    This is what the drift detector catches in Stage 9.

COLUMNS
-------
Numeric (count/duration):
  Administrative, Administrative_Duration    - pages/time on admin section
  Informational, Informational_Duration      - pages/time on info section
  ProductRelated, ProductRelated_Duration    - pages/time on products (HIGH signal)

Numeric (ratios):
  BounceRates   - fraction that left immediately (negatively correlated with purchase)
  ExitRates     - fraction that exited from each page (negatively correlated)
  PageValues    - average $ value of pages visited before purchase (STRONGEST predictor)
  SpecialDay    - proximity to a special day (0=none, 1=on the day)

Numeric (encoded categoricals treated as ordinal):
  OperatingSystems (1-8), Browser (1-13), Region (1-9), TrafficType (1-20)

Categorical:
  Month         - month of session (Feb-Oct for train, Nov-Dec for current)
  VisitorType   - Returning_Visitor / New_Visitor / Other
  Weekend       - True / False (str)

Target:
  Revenue       - 1 = session ended in purchase, 0 = did not
"""

import numpy as np
import pandas as pd


# ── Month distributions ───────────────────────────────────────────────────────
# UCI dataset: sessions are NOT uniformly distributed across months.
# May has the most non-holiday sessions; Feb has the fewest.
_TRAIN_MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
_TRAIN_MONTH_PROBS = [0.05, 0.09, 0.10, 0.18, 0.11, 0.07, 0.10, 0.15, 0.15]

_HOLIDAY_MONTHS = ["Nov", "Dec"]
_HOLIDAY_MONTH_PROBS = [0.55, 0.45]


def generate_shopper_data(
    n: int = 10000, holiday: bool = False, seed: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic online shoppers session DataFrame.

    Parameters
    ----------
    n       : number of sessions to generate
    holiday : if True, simulate Nov-Dec holiday distribution shift
              (higher PageValues, lower BounceRates, more SpecialDay)
    seed    : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with all columns matching the UCI Online Shoppers schema
    """
    rng = np.random.default_rng(seed)

    # ── technical / demographic attributes ───────────────────────────────────
    # These are stable across seasons (low drift velocity)
    os_type     = rng.integers(1, 9, n).astype(float)
    browser     = rng.integers(1, 14, n).astype(float)
    region      = rng.integers(1, 10, n).astype(float)
    traffic     = rng.integers(1, 21, n).astype(float)

    # Visitor type: 85% returning, 14% new, 1% other
    # Holiday shift: more new visitors (gift buyers, first-timers)
    if holiday:
        vtype_probs = [0.76, 0.22, 0.02]
    else:
        vtype_probs = [0.85, 0.14, 0.01]
    visitor_type = rng.choice(
        ["Returning_Visitor", "New_Visitor", "Other"], n, p=vtype_probs
    )

    # Weekend: ~28% of sessions happen on weekends
    weekend = rng.random(n) < 0.28

    # Month
    if holiday:
        month = rng.choice(_HOLIDAY_MONTHS, n, p=_HOLIDAY_MONTH_PROBS)
    else:
        month = rng.choice(_TRAIN_MONTHS, n, p=_TRAIN_MONTH_PROBS)

    # ── page visit counts (zero-inflated) ─────────────────────────────────────
    # Most sessions don't visit Administrative or Informational pages at all.
    # ProductRelated is visited by almost everyone (it's the product catalogue).
    admin_visits  = (rng.random(n) > 0.45) * rng.integers(0, 12, n).astype(float)
    info_visits   = (rng.random(n) > 0.60) * rng.integers(0, 8,  n).astype(float)
    prod_visits   = rng.integers(1, 80, n).astype(float)

    # Duration is correlated with visit count; zero if no visits
    admin_dur = np.where(admin_visits > 0, admin_visits * rng.exponential(60, n), 0.0)
    info_dur  = np.where(info_visits  > 0, info_visits  * rng.exponential(55, n), 0.0)
    prod_dur  = prod_visits * rng.exponential(90, n)

    # Holiday: more product browsing
    if holiday:
        prod_visits = np.clip(prod_visits * rng.uniform(1.15, 1.35, n), 1, 700).astype(float)
        prod_dur    = prod_visits * rng.exponential(95, n)

    # ── engagement quality metrics ─────────────────────────────────────────────
    # BounceRates: typically low (most users visit >1 page)
    # ExitRates: correlated with BounceRates but slightly higher
    # Holiday: engaged shoppers — both rates are lower
    if holiday:
        bounce = np.clip(rng.beta(1.2, 10, n), 0, 1)
        exit_r = np.clip(bounce + rng.normal(0.01, 0.02, n), 0, 1)
    else:
        bounce = np.clip(rng.beta(1.5, 8, n), 0, 1)
        exit_r = np.clip(bounce + rng.normal(0.02, 0.02, n), 0, 1)

    # ── PageValues — THE KEY PREDICTOR ────────────────────────────────────────
    # PageValues = avg dollar value of pages a session visited before purchase.
    # Non-buyers: overwhelmingly 0 (they never reached a pre-checkout page).
    # Buyers: substantial PageValues.
    # Holiday: distribution shifts up — users arrive with more purchase intent,
    #          so even non-buyers have higher PageValues (browsing gift pages).
    if holiday:
        pv_nonbuyer = rng.exponential(8, n)   # higher baseline in holiday
        pv_buyer    = rng.exponential(55, n)
    else:
        pv_nonbuyer = rng.exponential(3, n)
        pv_buyer    = rng.exponential(45, n)

    # ── SpecialDay proximity ──────────────────────────────────────────────────
    # 0 = far from any special day; 1 = it IS a special day
    # Most of the year: near-zero. Holiday season: many promotional days.
    if holiday:
        special_day = rng.choice(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n,
            p=[0.50, 0.12, 0.12, 0.10, 0.08, 0.08]
        )
    else:
        special_day = rng.choice(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n,
            p=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01]
        )

    # ── Purchase probability (Revenue) ───────────────────────────────────────
    # Risk factors derived from Sakar et al. (2019) Table 4 feature importances:
    #   PageValues       → strongest single predictor (feature importance ~0.35)
    #   ExitRates        → negative predictor (high exit → less likely to buy)
    #   ProductRelated   → more product browsing → more likely to buy
    #   BounceRates      → negative predictor
    #   VisitorType      → Returning visitors buy more
    #   SpecialDay       → promotional proximity increases purchase

    vtype_bonus = np.where(visitor_type == "Returning_Visitor", 0.4,
                  np.where(visitor_type == "New_Visitor",        0.1, 0.0))
    weekend_bonus = np.where(weekend, 0.1, 0.0)

    risk = (
        0.025 * pv_nonbuyer                    # baseline PageValues signal
        + 0.015 * np.log1p(prod_visits)        # browsing depth
        - 3.5   * exit_r                       # high exit = no buy
        - 3.0   * bounce                       # bounce = no buy
        + 0.8   * special_day                  # promotional lift
        + vtype_bonus
        + weekend_bonus
        + rng.normal(0, 1.0, n)               # irreducible noise
    )
    # Center = 0.5 calibrated so that ~15% of sessions are positive (Sakar et al. 2019).
    # With risk ~ N(-0.57, 1), P(risk > 0.5) ≈ 14-16% — matches the real dataset.
    prob = 1 / (1 + np.exp(-(risk - 0.5) / 1.5))

    # Assign PageValues: buyers get the buyer distribution; non-buyers the baseline
    revenue = (prob > 0.5).astype(int)
    page_values = np.where(revenue == 1, pv_buyer, pv_nonbuyer)
    page_values = np.clip(page_values, 0, 400)

    df = pd.DataFrame({
        "Administrative":             np.clip(admin_visits, 0, 30),
        "Administrative_Duration":    np.clip(admin_dur,    0, 3600),
        "Informational":              np.clip(info_visits,  0, 20),
        "Informational_Duration":     np.clip(info_dur,     0, 3600),
        "ProductRelated":             np.clip(prod_visits,  0, 700),
        "ProductRelated_Duration":    np.clip(prod_dur,     0, 63000),
        "BounceRates":                bounce,
        "ExitRates":                  exit_r,
        "PageValues":                 page_values,
        "SpecialDay":                 special_day.astype(float),
        "Month":                      month,
        "OperatingSystems":           os_type,
        "Browser":                    browser,
        "Region":                     region,
        "TrafficType":                traffic,
        "VisitorType":                visitor_type,
        "Weekend":                    weekend.astype(str),
        "Revenue":                    revenue,
    })

    return df
