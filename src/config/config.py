# =============================================================================
# config.py  -  Central Configuration
# =============================================================================
# MLOps Lesson: "Magic numbers scattered across files are a maintenance nightmare.
#                Every threshold must have a documented business reason — not just
#                a value someone liked the look of."
#
# BUSINESS CONTEXT
# ----------------
# Problem  : Predict whether an e-commerce session ends in a purchase (Revenue=1)
# Domain   : Online retail — 12,330 real user sessions from a Turkey-based retailer
# Data     : UCI Online Shoppers Purchasing Intention Dataset (Sakar et al. 2019)
# Target   : Will this browsing session result in a purchase? (1=yes, 0=no)
# Period   : One full year — sessions span Feb through Dec
#
# WHY THIS PROBLEM IS HARD
# -------------------------
# Class imbalance: only 15.5% of sessions end in a purchase.
# A naive classifier (always predict no-purchase) achieves 84.5% accuracy.
# Accuracy is completely USELESS here. F1 on the minority class is the real signal.
#
# Temporal drift: the holiday months (Nov-Dec) look very different from the
# rest of the year. Users arrive with higher purchase intent (PageValues spikes),
# bounce less, and visit more product pages. A model trained on Feb-Oct
# underperforms in Nov-Dec unless retrained with holiday-season data.
#
# WHY EVERY NUMBER HERE HAS A REASON
# ------------------------------------
# The business cost model drives every threshold:
#
#   False Negative (miss a buyer):
#     Retailer loses the conversion opportunity. Lost average order ≈ $85.
#     No targeted offer was served. No remarketing triggered.
#
#   False Positive (flag a non-buyer):
#     Wasted discount / promotional offer sent. Cost ≈ $3 per false alarm.
#
#   Cost ratio FN:FP ≈ 28:1
#   → Recall matters far more than precision.
#   → F1 on the positive class (Revenue=True) is the primary optimisation target.
# =============================================================================

import os

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "models"
LOG_DIR   = "logs"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Temporal split ────────────────────────────────────────────────────────────
# Training months: Feb–Oct (non-holiday browsing behaviour)
# Current batch  : Nov–Dec (holiday shopping season — the drift batch)
#
# WHY THIS SPLIT
# --------------
# The holiday season (Nov-Dec) is a fundamentally different distribution:
#   - Users arrive with higher purchase intent → PageValues spikes
#   - SpecialDay is non-zero for a larger fraction of sessions
#   - BounceRates and ExitRates drop (more engaged shoppers)
#   - More New_Visitors (gift buyers who don't regularly shop this site)
#
# Training on Feb-Oct and monitoring on Nov-Dec mimics a production scenario
# where a model trained in Q3 is deployed into the Q4 holiday season.
# Drift detection should catch this shift before performance visibly degrades.
TRAIN_MONTHS   = {"Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"}
CURRENT_MONTHS = {"Nov", "Dec"}

# ── Feature schema ────────────────────────────────────────────────────────────
# Source: Sakar et al. (2019), Table 1
#
# Each feature is annotated with its VELOCITY — how fast it legitimately changes.
# Velocity determines the per-feature drift threshold (see DRIFT_THRESHOLDS below).

NUMERIC_FEATURES = [
    # Page interaction counts — medium velocity (vary by campaign and season)
    "Administrative",              # pages visited in the Administrative category
    "Administrative_Duration",     # time spent on Administrative pages (seconds)
    "Informational",               # pages visited in the Informational category
    "Informational_Duration",      # time spent on Informational pages
    "ProductRelated",              # pages visited in the Product-related category (HIGH signal)
    "ProductRelated_Duration",     # time spent on product pages (HIGH signal)

    # Engagement quality metrics — key predictors, HIGH drift sensitivity
    "BounceRates",                 # fraction of sessions that left from entry page
    "ExitRates",                   # fraction of sessions that exited from this page
    "PageValues",                  # avg value of pages before completing a transaction (STRONGEST predictor)
    "SpecialDay",                  # proximity to a special shopping day (0–1; spikes in holidays)

    # Technical session attributes — low velocity
    "OperatingSystems",            # integer-coded OS (1–8)
    "Browser",                     # integer-coded browser (1–13)
    "Region",                      # integer-coded geographic region (1–9)
    "TrafficType",                 # integer-coded traffic source (1–20)
]

CATEGORICAL_FEATURES = [
    "Month",          # Jan–Dec; used for temporal slice evaluation
    "VisitorType",    # Returning_Visitor | New_Visitor | Other
    "Weekend",        # True | False (bool stored as string after CSV round-trip)
]

TARGET = "Revenue"

# ── Validation rules ──────────────────────────────────────────────────────────
# NULL_THRESHOLD: abort pipeline if more than this fraction of a column is null.
#
# WHY 8%: Unlike bank data, e-commerce clickstream data has legitimate gaps.
# Duration fields can be 0/null for bounce sessions. SpecialDay is sparse by design.
# 8% is the threshold beyond which missing values signal a tracking/instrumentation
# failure (e.g., the JavaScript event tracker is broken) rather than real nulls.
NULL_THRESHOLD = 0.08

VALID_VISITOR_TYPES = {"Returning_Visitor", "New_Visitor", "Other"}
VALID_MONTHS        = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
VALID_WEEKEND       = {"True", "False", True, False, 0, 1}

# Hard bounds — values outside these are data errors, not extreme sessions.
# Sources: UCI dataset documentation + Sakar et al. (2019) observed ranges.
COLUMN_RANGES = {
    "Administrative":             (0,    30),     # max reasonable admin pages per session
    "Administrative_Duration":    (0,  3600),     # max 1 hour on admin pages
    "Informational":              (0,    20),
    "Informational_Duration":     (0,  3600),
    "ProductRelated":             (0,   700),     # power shoppers browse hundreds of pages
    "ProductRelated_Duration":    (0, 63000),     # ~17.5 hours absolute max
    "BounceRates":                (0,     1),
    "ExitRates":                  (0,     1),
    "PageValues":                 (0,   400),     # dollars; outliers exist but cap at 400
    "SpecialDay":                 (0,     1),
    "OperatingSystems":           (1,     8),
    "Browser":                    (1,    13),
    "Region":                     (1,     9),
    "TrafficType":                (1,    20),
}

# ── Evaluation gates ──────────────────────────────────────────────────────────
# CRITICAL: 15.5% positive rate.
# Naive classifier (always 0) = 84.5% accuracy — USELESS.
# All gates are evaluated on the HELD-OUT TEST SET (20% of training months).
#
# PRIMARY METRIC: F1 on minority class (Revenue=True)
# Why F1 not AUC: The business cares about actual predictions, not rankings.
# A model that classifies correctly at the deployment threshold matters more
# than one with a great ROC curve but poor F1 at threshold=0.5.
#
# Gate derivations:
#
# F1_THRESHOLD = 0.65  ← PRIMARY GATE (minority class F1)
#   LightGBM with balanced weights reliably achieves 0.72–0.80 on this dataset.
#   A tuned RandomForest typically gives 0.48–0.58 — below this gate.
#   0.65 means the model has genuinely learned to identify buyers, not just
#   benefiting from class imbalance. The naive baseline gets ~0.00 F1 on the
#   positive class (it never predicts positive).
#
# RECALL_THRESHOLD = 0.60
#   At 28:1 FN:FP cost ratio, recall is the primary business lever.
#   60% recall floor: model must catch at least 6 in 10 actual buyers.
#   Below 0.60, the model is less effective than a simple "show everyone an offer".
#
# PRECISION_THRESHOLD = 0.60
#   60% precision means at most 40% of flagged sessions are false alarms.
#   At $3/false alarm and $85/missed buyer, this is well within acceptable range.
#   Setting too low causes promotional budget waste.
#
# AUC_ROC_THRESHOLD = 0.85
#   LightGBM typically achieves 0.88–0.92 AUC-ROC on this dataset.
#   Below 0.85: the model cannot meaningfully rank sessions by purchase probability,
#   which breaks personalisation, bid optimisation, and remarketing pipelines.
#
# AUC_PR_THRESHOLD = 0.60
#   PR-AUC is the right metric for imbalanced classification.
#   Random classifier PR-AUC = positive rate = 0.155.
#   0.60 means the model has substantial lift over random across all thresholds.

F1_THRESHOLD        = 0.65   # minority class F1 — PRIMARY GATE
RECALL_THRESHOLD    = 0.60
PRECISION_THRESHOLD = 0.60
AUC_ROC_THRESHOLD   = 0.85
AUC_PR_THRESHOLD    = 0.60

# ── Drift detection — per-feature velocity thresholds ────────────────────────
# WHY DIFFERENT THRESHOLDS PER FEATURE
# --------------------------------------
# PageValues (the strongest predictor) has HIGH seasonal velocity:
#   - Normal months: mean ~5–8. Holiday months: mean ~15–25.
#   - A 15% shift in PageValues in a non-seasonal comparison is alarming.
#   - But Nov-Dec vs Feb-Oct shift is ~80–100%. We set the threshold TIGHT
#     for high-importance features so holiday drift is caught early.
#
# BounceRates and ExitRates: LOW velocity (small day-to-day changes).
#   Any shift > 10% is meaningful since these are normalised 0-1 ratios.
#
# Technical features (Browser, Region, TrafficType): LOW velocity.
#   These change slowly as user demographics evolve. Any shift > 8%
#   is likely a data engineering issue, not organic behaviour change.

DRIFT_THRESHOLDS = {
    # Very tight: strong predictors — any shift triggers retraining consideration
    "PageValues":              0.20,  # most important feature; holiday spike is huge
    "BounceRates":             0.10,  # engagement quality; stable in non-holiday
    "ExitRates":               0.10,  # ditto

    # Medium: product browsing activity — seasonal but not extreme
    "ProductRelated":          0.20,
    "ProductRelated_Duration": 0.25,
    "SpecialDay":              0.25,  # by design near-zero most months, spikes in Nov-Dec

    # Medium-low: administrative/informational pages change with site structure
    "Administrative":          0.25,
    "Administrative_Duration": 0.30,
    "Informational":           0.25,
    "Informational_Duration":  0.30,

    # Low velocity: technical attributes are stable
    "OperatingSystems":        0.08,
    "Browser":                 0.08,
    "Region":                  0.08,
    "TrafficType":             0.12,
}

# Global fallback for any feature not listed
DRIFT_THRESHOLD = 0.15

# ── High-importance features for Rule 1 retrain trigger ──────────────────────
# These features drive the majority of model variance.
# If ANY of these drift, Rule 1 fires regardless of other signals.
# WHY: PageValues alone explains >35% of feature importance in LightGBM.
#      A drift in PageValues means the model's strongest signal has shifted.
RETRAIN_HIGH_IMPORTANCE_FEATURES = [
    "PageValues",       # strongest predictor — avg transaction value signal
    "BounceRates",      # user engagement quality
    "ExitRates",        # user engagement quality
    "ProductRelated",   # browsing depth on product pages
]

# ── Retrain trigger — multi-rule logic ───────────────────────────────────────
# Rule 2: retrain if >= RETRAIN_MIN_FEATURES features exceed their threshold
RETRAIN_MIN_FEATURES = 2

# Rule 3: retrain if F1 on current batch drops > F1_DEGRADATION_THRESHOLD
# WHY 0.05: A 5pp F1 drop = ~7 additional missed buyers per 1,000 sessions.
# At $85/missed conversion, that's ~$595/day at typical traffic volumes.
# This is material enough to justify retraining.
F1_DEGRADATION_THRESHOLD = 0.05

# Rule 4: minimum interval between retrains (days)
# WHY 14 (vs 30 for credit cards): E-commerce is faster-moving than credit risk.
# Marketing campaigns run weekly. Holiday season lasts 6-8 weeks.
# A 14-day minimum prevents over-reaction to daily noise while allowing
# rapid response to genuine seasonal shifts.
MIN_RETRAIN_INTERVAL_DAYS = 14
