import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))
    

from paths import INTERMEDIATE_CSV_DIR, WINDOWS_DIR

# Unified canonical label mapping
LABEL_MAPPING = {
    # Room A classes
    "fall": "FALL",
    "sit": "SIT",
    "sitdown": "SIT_DOWN",
    "stand": "STAND",
    "standup": "STAND_UP",
    "walk": "WALK",
    # Room B classes
    "empty": "EMPTY",
    "lying": "LYING",
    "sitting": "SIT",
    "standing": "STAND",
    "walking": "WALK",
}

def extract_and_normalize_window(df: pd.DataFrame, target_len: int = 150) -> np.ndarray:
    """
    Extracts a fixed (target_len, 256) float32 window and applies 
    temporal z-score standardization per subcarrier.
    """
    # Extract numerical matrix (N, 256)
    data = df.iloc[:, :256].to_numpy(dtype=np.float32)
    n_rows, n_subcarriers = data.shape

    if n_rows >= target_len:
        start_idx = (n_rows - target_len) // 2
        window = data[start_idx : start_idx + target_len, :]
    else:
        pad_before = (target_len - n_rows) // 2
        pad_after = target_len - n_rows - pad_before
        window = np.pad(data, ((pad_before, pad_after), (0, 0)), mode="edge")

    # Temporal standardization per subcarrier: (x - mean) / (std + eps)
    mean = np.mean(window, axis=0, keepdims=True)
    std = np.std(window, axis=0, keepdims=True)
    normalized_window = (window - mean) / (std + 1e-7)

    return normalized_window.astype(np.float32)

def generate_windows_for_room(
    intermediate_dir: Path,
    output_dir: Path,
    room_name: str,
    target_len: int = 150
):
    csv_files = list(intermediate_dir.rglob("*.csv"))
    # Exclude previously saved manifest files
    csv_files = [f for f in csv_files if not f.name.endswith("_manifest.csv")]
    
    print(f"[{room_name}] Found {len(csv_files)} intermediate CSV captures.")
    records = []

    for idx, csv_path in enumerate(csv_files, 1):
        rel_parts = csv_path.relative_to(intermediate_dir).parts
        # Expected structure: <split>/<raw_activity>/<filename.csv>
        split = rel_parts[0]
        raw_activity = rel_parts[1].lower()
        canonical_label = LABEL_MAPPING.get(raw_activity, raw_activity.upper())

        try:
            df = pd.read_csv(csv_path)
            if df.empty or len(df) == 0:
                continue

            window = extract_and_normalize_window(df, target_len=target_len)

            # File naming: window_<room>_<split>_<id>.npy
            window_id = f"win_{room_name}_{split}_{idx:04d}"
            capture_id = f"capture_{room_name}_{split}_{idx:04d}"
            save_folder = output_dir / split / room_name
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / f"{window_id}.npy"

            np.save(save_path, window)

            records.append({
                "window_id": window_id,
                "capture_id": capture_id,  # NEW
                "room": room_name,
                "split": split,
                "activity": canonical_label,
                "original_activity": raw_activity,
                "original_rows": len(df),
                "window_shape": str(window.shape),
                "file_path": save_path.relative_to(Path(__file__).resolve().parent.parent).as_posix(),
                "source_csv": csv_path.relative_to(Path(__file__).resolve().parent.parent).as_posix()
            })
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")

    return records

def build_benchmark_windows():
    room_a_dir = INTERMEDIATE_CSV_DIR / "room_A"
    room_b_dir = INTERMEDIATE_CSV_DIR / "room_B"
    output_windows_dir = WINDOWS_DIR
    output_windows_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # Process Room A captures
    records_a = generate_windows_for_room(room_a_dir, output_windows_dir, room_name="Room_A")
    all_records.extend(records_a)

    # Process Room B captures
    records_b = generate_windows_for_room(room_b_dir, output_windows_dir, room_name="Room_B")
    all_records.extend(records_b)

    # Export master windows manifest
    manifest_df = pd.DataFrame(all_records)
    master_manifest_path = output_windows_dir / "windows_manifest.csv"
    manifest_df.to_csv(master_manifest_path, index=False)

    print("\n" + "=" * 65)
    print("Window Generation Complete!")
    print(f"Master manifest saved to: {master_manifest_path.resolve()}")
    print("\nWindow count breakdown by Split, Room, and Activity:")
    summary = manifest_df.groupby(["split", "room", "activity"])["window_id"].count()
    print(summary)
    print("=" * 65)

if __name__ == "__main__":
    build_benchmark_windows()