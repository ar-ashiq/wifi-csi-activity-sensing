
from pathlib import Path

# Import processing functions
from parser_experiment2 import process_experiment2_folders
from process_experiment3 import process_room_b
from build_windows import build_benchmark_windows

# Import paths from your configuration module
from paths import (
    EXPERIMENT2_DATA_DIR,
    ROOM_A_CSV_DIR,
    EXPERIMENT3_DATA_DIR,
    ROOM_B_CSV_DIR,
)


def run_experiment2():
    print("\n========== STEP 1: PARSE EXPERIMENT 2 ==========")

    input_path = EXPERIMENT2_DATA_DIR
    output_path = ROOM_A_CSV_DIR

    process_experiment2_folders(input_path, output_path)

    print("Experiment 2 parsing completed.")


def run_experiment3():
    print("\n========== STEP 2: PROCESS EXPERIMENT 3 ==========")

    possible_paths = [
        EXPERIMENT3_DATA_DIR / "Data",
        EXPERIMENT3_DATA_DIR,
    ]

    exp3_data_path = None

    for path in possible_paths:
        if path.exists() and any(path.rglob("*.csv")):
            exp3_data_path = path
            break

    if exp3_data_path is None:
        exp3_data_path = EXPERIMENT3_DATA_DIR / "Data"

    output_path = ROOM_B_CSV_DIR

    process_room_b(
        str(exp3_data_path),
        str(output_path),
        seed=42
    )

    print("Experiment 3 processing completed.")


def run_build_windows():
    print("\n========== STEP 3: BUILD WINDOWS ==========")

    build_benchmark_windows()

    print("Window generation completed.")


def main():
    print("\nStarting complete data processing pipeline...\n")

    run_experiment2()

    run_experiment3()

    run_build_windows()

    print("\n========== PIPELINE COMPLETED SUCCESSFULLY ==========")


if __name__ == "__main__":
    main()
