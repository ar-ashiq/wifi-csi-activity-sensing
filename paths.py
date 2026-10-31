"""Shared project paths used by the application."""

from importlib.resources import path
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT

# SRC = PROJECT_ROOT / "src"
SRC = PROJECT_ROOT.joinpath("src")
RAW_DATASET_DIR = PROJECT_ROOT.joinpath("raw")


INTERMEDIATE_CSV_DIR = PROJECT_ROOT.joinpath("intermediate_csv")

CONFIGS_DIR = PROJECT_ROOT.joinpath("configs")
WINDOWS_DIR = PROJECT_ROOT.joinpath("windows")
CHECKPOINTS_DIR = PROJECT_ROOT.joinpath("checkpoints")

ROOM_A_CSV_DIR = PROJECT_ROOT.joinpath("intermediate_csv", "room_A")
ROOM_B_CSV_DIR = PROJECT_ROOT.joinpath("intermediate_csv", "room_B")
EXPERIMENT2_DATA_DIR = PROJECT_ROOT.joinpath("raw", "Experiment-2", "realdata", "input_data", "30ms", "csidata" )
EXPERIMENT3_DATA_DIR = PROJECT_ROOT.joinpath("raw", "Experiment-3", "Data")

# files
TASK1_ABLATION_CSV = CHECKPOINTS_DIR.joinpath("task1_window_ablation.csv")