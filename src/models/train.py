"""
train.py  -  Model Training
=============================
MLOps Lesson: "A training script is not a notebook cell — it must be
               deterministic, parameterised, and callable by a pipeline."

WHY LIGHTGBM (not RandomForest)
---------------------------------
This is an end-of-ML-program dataset. RandomForest is a good baseline
but cannot beat ~58-62% F1 on the minority class here because:

  1. Class imbalance (15.5% positive): RF with default settings is biased
     toward the majority class. It needs class_weight or subsampling tricks.

  2. Feature interactions: PageValues × BounceRates, SpecialDay × Month
     are complex non-linear interactions that RF captures poorly without
     very deep trees (which causes overfitting).

  3. Gradient boosting (LightGBM) handles imbalance via scale_pos_weight,
     learns feature interactions through sequential residual fitting, and
     typically achieves 0.72–0.80 F1 on this dataset vs 0.48–0.62 for RF.

WHY scale_pos_weight (not SMOTE)
----------------------------------
SMOTE generates synthetic minority samples. This:
  - Slows training (more data)
  - Risks overfitting near class boundaries
  - Doesn't help at inference time (you can't generate synthetic buyers)

scale_pos_weight = (n_negative / n_positive) directly tells LightGBM
to penalise false negatives proportionally to their business cost.
At 15.5% positive rate: scale_pos_weight ≈ 5.45.

WHY NOT TUNE HYPERPARAMETERS HERE
-----------------------------------
Hyperparameter tuning (Optuna, GridSearchCV) belongs in retrain.py's
'reexperiment' strategy, not in the production training script.
The production script trains with CHOSEN hyperparameters.
Those choices are logged to experiments.csv so they can be reviewed.

WHY RANDOM_STATE IS NON-NEGOTIABLE
------------------------------------
Without a fixed seed, two runs on identical data produce different models
with slightly different metrics. "Did the metrics change because of my
code change or because of randomness?" — the seed pins the answer.
"""

import numpy as np
import lightgbm as lgb
from src.config.config import SEED


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict = None,
) -> lgb.LGBMClassifier:
    """
    Fit a LightGBM classifier with fixed hyperparameters and balanced class weights.

    Parameters
    ----------
    X      : np.ndarray  (n_samples, n_features)
    y      : np.ndarray  (n_samples,)  – binary labels
    params : optional dict of hyperparameter overrides
             (used by retrain.py's reexperiment strategy)

    Returns
    -------
    Fitted LGBMClassifier

    Expected performance on UCI Online Shoppers (test set, no tuning):
      F1 (minority class) : 0.72 – 0.80
      AUC-ROC             : 0.88 – 0.92
      AUC-PR              : 0.62 – 0.72
    """
    n_negative  = int((y == 0).sum())
    n_positive  = int((y == 1).sum())
    pos_rate    = n_positive / len(y)
    scale_w     = n_negative / max(n_positive, 1)

    print(f"  Class distribution: {n_positive:,} positive ({pos_rate:.1%}), "
          f"{n_negative:,} negative")
    print(f"  scale_pos_weight = {scale_w:.2f}  "
          f"(penalises false negatives {scale_w:.1f}x vs false positives)")

    default_params = dict(
        n_estimators       = 500,
        learning_rate      = 0.05,
        num_leaves         = 63,        # controls tree complexity; 63 is a good default
        min_child_samples  = 20,        # prevent leaves with too few samples (overfitting)
        subsample          = 0.8,       # row subsampling — reduces overfitting
        colsample_bytree   = 0.8,       # feature subsampling per tree
        scale_pos_weight   = scale_w,   # handles class imbalance
        random_state       = SEED,
        n_jobs             = -1,
        verbose            = -1,        # suppress LightGBM's per-iteration output
    )

    if params:
        default_params.update(params)
        print(f"  Hyperparameter overrides: {params}")

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X, y)

    print(f"  Trained LGBMClassifier: {model.n_estimators} trees, "
          f"lr={model.learning_rate}, num_leaves={model.num_leaves}")
    print(f"  Training set: {X.shape[0]:,} samples x {X.shape[1]} features")

    return model
