"""
reproducibility.py  –  Seed Management
==========================================
MLOps Lesson: "Reproducibility is not a nice-to-have – it is the
               foundation of debugging."

WHY SEEDS ARE NOT ENOUGH
--------------------------
Setting a seed makes your Python process deterministic.  But the same
code on a different machine can still produce different results because:

  * BLAS/LAPACK libraries (used by numpy) may use different thread counts
    and have non-deterministic floating-point accumulation order.
  * Different numpy/sklearn versions implement algorithms differently.
  * GPU operations are not deterministic by default (different issue,
    but important for deep learning pipelines).

True reproducibility requires pinning:
  1. The random seed (done here)
  2. The package versions (see requirements.txt)
  3. The OS/hardware (Docker containers or reproducible build systems)

For tree-based models (RandomForest), seeds + pinned packages are
sufficient.  For neural networks on GPUs, it's much harder.
"""

import random
import numpy as np

from src.config.config import SEED


def set_seed(seed: int = SEED):
    """
    Set random seeds for the Python standard library and NumPy.

    Note: sklearn models use their own random_state parameter (set in
    train.py).  This function handles the global NumPy state that affects
    data generation and any other random operations in the pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"  Random seed set to {seed}")
