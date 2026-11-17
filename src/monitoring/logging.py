"""
monitoring/logging.py  –  Prediction & Latency Logging
=========================================================
MLOps Lesson: "You cannot monitor what you do not log."

WHY LOG PREDICTIONS
-------------------
Logging every prediction enables:
  1. Outcome tracking: when purchase ground-truth labels arrive (after
     the session ends), you can join them to predictions and compute
     live accuracy and recall.
  2. Distribution monitoring: if the fraction of high-intent predictions
     suddenly spikes or drops, something has changed upstream.
  3. Auditability: being able to explain "why was this session flagged
     as high-intent?" requires a record of the input and output.

WHY LOG LATENCY
---------------
Latency creep is a silent production problem.  A model that took 20ms
in testing might take 200ms in production (larger input batches, cold
starts, I/O waits).  Without latency logs, you only discover this when
users complain.

WHY THIS APPROACH DOES NOT SCALE
---------------------------------
Writing to a local CSV file works for one server.  When you have:
  * 10 prediction servers running in parallel
  * Millions of predictions per day
  * A need for real-time dashboards

...you need a streaming log system: Kafka → ClickHouse, or a managed
observability platform (Datadog, Grafana + Loki, AWS CloudWatch).
The local CSV is a conceptual placeholder.  The important idea is:
"every prediction must be logged somewhere queryable."
"""

import os
import time
import datetime

import numpy as np
import pandas as pd

from src.config.config import LOG_DIR


def log_predictions(preds: np.ndarray):
    """
    Append binary predictions to logs/predictions.csv.

    One row per prediction with a timestamp.
    This enables later joining with ground-truth labels once they arrive.
    """
    pred_log = pd.DataFrame({
        "prediction": preds.tolist(),
        "timestamp":  str(datetime.datetime.now()),
    })

    pred_path = os.path.join(LOG_DIR, "predictions.csv")
    pred_log.to_csv(
        pred_path,
        mode   = "a",
        index  = False,
        header = not os.path.exists(pred_path),
    )

    pos_rate = preds.mean()
    print(f"  Logged {len(preds):,} predictions  |  positive rate: {pos_rate:.1%}")


def log_latency(start_time: float):
    """
    Append inference latency to logs/latency.csv.

    Parameters
    ----------
    start_time : time.time() value captured just before model.predict() was called
    """
    latency_s  = time.time() - start_time
    latency_ms = latency_s * 1000

    lat_path = os.path.join(LOG_DIR, "latency.csv")
    with open(lat_path, "a") as f:
        f.write(f"{latency_ms:.4f},{datetime.datetime.now()}\n")

    print(f"  Inference latency: {latency_ms:.3f} ms")
