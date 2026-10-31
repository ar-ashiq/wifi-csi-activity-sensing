"""Shared project paths used by the application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT
SRC = PROJECT_ROOT / "src"
RAW_DATASET_DIR = PROJECT_ROOT / "raw"
INTERMEDIATE_CSV_DIR = PROJECT_ROOT / "intermediate_csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"
WINDOWS_DIR = PROJECT_ROOT / "windows"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

ROOM_A_CSV_DIR = INTERMEDIATE_CSV_DIR / "room_A"
ROOM_B_CSV_DIR = INTERMEDIATE_CSV_DIR / "room_B"
EXPERIMENT2_DATA_DIR = RAW_DATASET_DIR / "experiment_2" / "csidata"
EXPERIMENT3_DATA_DIR = RAW_DATASET_DIR / "experiment_3"