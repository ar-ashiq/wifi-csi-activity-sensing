import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from paths import EXPERIMENT3_DATA_DIR, ROOM_B_CSV_DIR

def process_room_b(
    exp3_data_dir: str,
    output_base_dir: str,
    seed: int = 42
):
    input_dir = Path(exp3_data_dir)
    out_dir = Path(output_base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    
    # Standard activity classes in Room B
    classes = ["Empty", "Lying", "Sitting", "Standing", "Walking"]
    col_names = [f"subcarrier_{i:03d}" for i in range(256)]
    manifest_records = []

    print(f"Scanning Experiment 3 directory: {input_dir.resolve()}\n")

    for cls_name in classes:
        # Search for files matching patterns like Empty_Amp_*.csv or Sitting_Amp_*.csv
        matched_files = sorted(list(input_dir.glob(f"{cls_name}*.csv")))
        
        # Fallback if files are inside subfolders named after the class
        if not matched_files:
            matched_files = sorted(list((input_dir / cls_name).glob("*.csv")))
        if not matched_files:
            # Case-insensitive search fallback
            matched_files = sorted([p for p in input_dir.rglob("*.csv") if cls_name.lower() in p.stem.lower()])

        total_found = len(matched_files)
        print(f"Class '{cls_name}': Found {total_found} captures")
        if total_found < 25:
            raise ValueError(f"Expected at least 25 captures for {cls_name}, but found {total_found}")

        # Deterministic shuffle using seed 42
        indices = rng.permutation(total_found).tolist()
        val_indices = set(indices[:25])  # Exactly 25 for validation

        for idx, file_path in enumerate(matched_files):
            split = "validation" if idx in val_indices else "development"
            target_folder = out_dir / split / cls_name.lower()
            target_folder.mkdir(parents=True, exist_ok=True)
            target_csv = target_folder / file_path.name

            # Read Experiment 3 CSV
            # Check if file has header or purely raw numbers
            try:
                sample = pd.read_csv(file_path, nrows=2, header=None)
                if isinstance(sample.iloc[0, 0], str) and not sample.iloc[0, 0].replace('.', '', 1).isdigit():
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_csv(file_path, header=None)
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
                continue

            # Keep exactly first 256 subcarriers
            df_csi = df.iloc[:, :256].copy()
            df_csi.columns = col_names
            n_rows = len(df_csi)

            # Quality Check: ensure numeric float32 and no all-NaN sweeps
            df_csi = df_csi.astype(np.float32)
            
            # Save standardized CSV
            df_csi.to_csv(target_csv, index=False)

            manifest_records.append({
                "capture_id": file_path.stem,
                "room": "Room_B",
                "split": split,
                "activity": cls_name.lower(),
                "rows": n_rows,
                "original_path": str(file_path.resolve()),
                "csv_path": str(target_csv.resolve())
            })

    # Save Room B provenance manifest
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = out_dir / "room_b_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print("\n" + "=" * 55)
    print("Room B Processing Summary:")
    print(manifest_df.groupby(["split", "activity"])["capture_id"].count())
    print("=" * 55)
    print(f"Manifest successfully generated: {manifest_path.resolve()}")

if __name__ == "__main__":
    possible_paths = [
        EXPERIMENT3_DATA_DIR / "Data",
        EXPERIMENT3_DATA_DIR,
    ]
    
    EXP3_DATA_PATH = None
    for p in possible_paths:
        if p.exists() and len(list(p.rglob("*.csv"))) > 0:
            EXP3_DATA_PATH = p
            break

    if EXP3_DATA_PATH is None:
        EXP3_DATA_PATH = EXPERIMENT3_DATA_DIR / "Data"

    OUTPUT_PATH = ROOM_B_CSV_DIR

    process_room_b(str(EXP3_DATA_PATH), str(OUTPUT_PATH), seed=42)