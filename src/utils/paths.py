"""
paths.py  –  Directory Initialisation
========================================
MLOps Lesson: "A pipeline that fails on the first run because a folder
               doesn't exist is a pipeline nobody will trust."

Creating required directories at startup is a small but important
reliability pattern.  It means the pipeline can run on a fresh machine
(or a fresh Docker container) without a README step saying "first run
mkdir -p data models logs".
"""

import os

from src.config.config import DATA_DIR, MODEL_DIR, LOG_DIR


def create_directories():
    """Create data/, models/, and logs/ directories if they don't exist."""
    for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"  Directories ready: {DATA_DIR}/  {MODEL_DIR}/  {LOG_DIR}/")
