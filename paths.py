"""Shared project paths used by the application."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = os.path.join(PROJECT_ROOT, "src")
RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "raw")
INTERMEDIATE_CSV_DIR = os.path.join(PROJECT_ROOT, "intermediate_csv")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
WINDOWS_DIR = os.path.join(PROJECT_ROOT, "windows")