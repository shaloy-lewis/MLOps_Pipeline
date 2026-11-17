"""
predict.py  –  Prediction Wrapper
===================================
MLOps Lesson: "Keep your predict function separate from your evaluation
               function.  At inference time you call predict; you never call
               evaluate."

WHY THIS IS A SEPARATE MODULE
-------------------------------
Keeping predict() isolated means the serving layer and the batch inference
script can import it without pulling in sklearn.metrics, gates, or
evaluation logic.  Separation of concerns keeps the dependency graph clean.

In a more complex system this module would also handle:
  * Probability calibration  (if the model's probabilities are not well-calibrated)
  * Threshold tuning         (predict_class() with a custom threshold, not 0.5)
  * Post-processing          (mapping class indices back to human-readable labels)
"""

import numpy as np


def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Return binary predictions (0 or 1) for a feature matrix X.

    Parameters
    ----------
    model : fitted sklearn model
    X     : np.ndarray  (n_samples, n_features)

    Returns
    -------
    np.ndarray of int  (n_samples,)
    """
    return model.predict(X)


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Return the probability of the positive class (purchased=1).

    Used by serve.py and batch_infer.py for risk stratification.

    Returns
    -------
    np.ndarray of float  (n_samples,)  – values in [0, 1]
    """
    return model.predict_proba(X)[:, 1]
