import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from paths import CHECKPOINTS_DIR, PROJECT_ROOT, WINDOWS_DIR

# Enforce deterministic GPU performance
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# 1. Dataset & Encoder Backbone
# ---------------------------------------------------------
class CSIWindowDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame):
        self.records = manifest_df.reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        file_path = Path(row["file_path"])
        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path
        data = np.load(file_path).astype(np.float32)  # (150, 256)
        tensor_x = torch.from_numpy(data.T)                  # (256, 150)
        return tensor_x, row["activity"], row["room"], row["window_id"]

class CSIEncoderBackbone(nn.Module):
    def __init__(self, in_channels: int = 256, embedding_dim: int = 128):
        super(CSIEncoderBackbone, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        feat = self.conv_blocks(x)
        return feat.view(feat.size(0), -1)

def extract_features(model, dataloader, device):
    """Passes windows through frozen backbone and outputs L2-normalized embeddings."""
    model.eval()
    embeddings = []
    activities = []
    rooms = []
    window_ids = []

    with torch.no_grad():
        for inputs, acts, rms, w_ids in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            feats = model(inputs)
            feats = F.normalize(feats, p=2, dim=-1)
            embeddings.append(feats.cpu().numpy())
            activities.extend(acts)
            rooms.extend(rms)
            window_ids.extend(w_ids)

    return np.vstack(embeddings), np.array(activities), np.array(rooms), np.array(window_ids)

# ---------------------------------------------------------
# 2. Episodic Few-Shot Adapter
# ---------------------------------------------------------
def run_adaptation_direction(
    source_room: str,
    target_room: str,
    target_exclusive_classes: list,
    shared_classes: list,
    dev_feats: np.ndarray,
    dev_acts: np.ndarray,
    dev_rooms: np.ndarray,
    val_feats: np.ndarray,
    val_acts: np.ndarray,
    val_rooms: np.ndarray,
    k_values: list = [1, 5, 10, 20],
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    results = []
    cm_k5 = None
    all_target_classes = shared_classes + target_exclusive_classes

    # Source prototypes for shared classes
    src_mask = dev_rooms == source_room
    src_feats = dev_feats[src_mask]
    src_acts = dev_acts[src_mask]

    shared_prototypes = {}
    for c in shared_classes:
        c_feats = src_feats[src_acts == c]
        if len(c_feats) > 0:
            shared_prototypes[c] = np.mean(c_feats, axis=0)
            shared_prototypes[c] /= np.linalg.norm(shared_prototypes[c]) + 1e-7

    # Target validation set
    tgt_val_mask = (val_rooms == target_room) & np.isin(val_acts, all_target_classes)
    X_val_target = val_feats[tgt_val_mask]
    y_val_target = val_acts[tgt_val_mask]

    # Target development candidates for exclusive support sampling
    tgt_dev_mask = dev_rooms == target_room
    tgt_dev_feats = dev_feats[tgt_dev_mask]
    tgt_dev_acts = dev_acts[tgt_dev_mask]

    for k in k_values:
        # Sample K distinct development captures per target-exclusive activity
        target_prototypes = {}
        for c in target_exclusive_classes:
            c_indices = np.where(tgt_dev_acts == c)[0]
            if len(c_indices) < k:
                sampled_indices = c_indices
            else:
                sampled_indices = rng.choice(c_indices, size=k, replace=False)
            
            c_feats = tgt_dev_feats[sampled_indices]
            target_prototypes[c] = np.mean(c_feats, axis=0)
            target_prototypes[c] /= np.linalg.norm(target_prototypes[c]) + 1e-7

        # Combine all prototypes
        active_classes = list(shared_prototypes.keys()) + list(target_prototypes.keys())
        prototype_matrix = np.stack(
            [shared_prototypes[c] if c in shared_prototypes else target_prototypes[c] for c in active_classes],
            axis=0
        )

        # Nearest-prototype inference via cosine similarity
        similarities = np.dot(X_val_target, prototype_matrix.T)
        pred_indices = np.argmax(similarities, axis=1)
        y_pred = np.array([active_classes[i] for i in pred_indices])

        # Overall Macro-F1 and Balanced Accuracy
        macro_f1 = f1_score(y_val_target, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_val_target, y_pred)

        # Target-Exclusive Macro-F1
        exclusive_mask = np.isin(y_val_target, target_exclusive_classes)
        if np.any(exclusive_mask):
            excl_macro_f1 = f1_score(y_val_target[exclusive_mask], y_pred[exclusive_mask], average="macro", zero_division=0)
        else:
            excl_macro_f1 = 0.0

        # Shared Activities Macro-F1
        shared_mask = np.isin(y_val_target, shared_classes)
        if np.any(shared_mask):
            shared_macro_f1 = f1_score(y_val_target[shared_mask], y_pred[shared_mask], average="macro", zero_division=0)
        else:
            shared_macro_f1 = 0.0

        results.append({
            "direction": f"{source_room}->{target_room}",
            "target_room": target_room,
            "K": k,
            "overall_macro_f1": round(macro_f1, 4),
            "exclusive_macro_f1": round(excl_macro_f1, 4),
            "shared_macro_f1": round(shared_macro_f1, 4),
            "balanced_accuracy": round(bal_acc * 100, 2)
        })

        if k == 5:
            cm = confusion_matrix(y_val_target, y_pred, labels=active_classes)
            cm_k5 = {
                "labels": active_classes,
                "matrix": cm.tolist(),
                "report": classification_report(y_val_target, y_pred, labels=active_classes, zero_division=0, output_dict=True)
            }

    return results, cm_k5

# ---------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------
def run_task3():
    manifest_path = WINDOWS_DIR / "windows_manifest.csv"
    checkpoints_dir = CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task 3 Running on: {device}")

    # Load frozen SSL backbone
    backbone = CSIEncoderBackbone(in_channels=256, embedding_dim=128).to(device)
    backbone_weights = checkpoints_dir / "ssl_encoder.pth"
    if not backbone_weights.exists():
        raise FileNotFoundError(f"Missing {backbone_weights}. Please run Task 2 first.")
    backbone.load_state_dict(torch.load(backbone_weights, map_location=device))
    backbone.eval()

    df_manifest = pd.read_csv(manifest_path)
    df_dev = df_manifest[df_manifest["split"] == "development"].copy()
    df_val = df_manifest[df_manifest["split"] == "validation"].copy()

    loader_kwargs = {"batch_size": 64, "shuffle": False, "pin_memory": (device.type == "cuda")}
    dev_dataset = CSIWindowDataset(df_dev)
    val_dataset = CSIWindowDataset(df_val)

    dev_loader = DataLoader(dev_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    print("Extracting L2-normalized representations across all splits on GPU...")
    dev_feats, dev_acts, dev_rooms, _ = extract_features(backbone, dev_loader, device)
    val_feats, val_acts, val_rooms, _ = extract_features(backbone, val_loader, device)

    # Class definitions per specification
    shared_classes = ["SIT", "STAND", "WALK"]
    room_a_exclusive = ["FALL", "SIT_DOWN", "STAND_UP"]
    room_b_exclusive = ["EMPTY", "LYING"]
    k_vals = [1, 5, 10, 20]

    all_adaptation_results = []

    # 1. Transfer Direction: Room A -> Room B
    print("\n" + "=" * 65)
    print("Evaluating Adaptation: Room A -> Room B")
    print("Source: Room A (Full Dev) | Target: Room B (K support for EMPTY, LYING)")
    print("=" * 65)
    res_a_to_b, cm_b_k5 = run_adaptation_direction(
        source_room="Room_A",
        target_room="Room_B",
        target_exclusive_classes=room_b_exclusive,
        shared_classes=shared_classes,
        dev_feats=dev_feats, dev_acts=dev_acts, dev_rooms=dev_rooms,
        val_feats=val_feats, val_acts=val_acts, val_rooms=val_rooms,
        k_values=k_vals
    )
    all_adaptation_results.extend(res_a_to_b)

    # 2. Transfer Direction: Room B -> Room A
    print("\n" + "=" * 65)
    print("Evaluating Adaptation: Room B -> Room A")
    print("Source: Room B (Full Dev) | Target: Room A (K support for FALL, SIT_DOWN, STAND_UP)")
    print("=" * 65)
    res_b_to_a, cm_a_k5 = run_adaptation_direction(
        source_room="Room_B",
        target_room="Room_A",
        target_exclusive_classes=room_a_exclusive,
        shared_classes=shared_classes,
        dev_feats=dev_feats, dev_acts=dev_acts, dev_rooms=dev_rooms,
        val_feats=val_feats, val_acts=val_acts, val_rooms=val_rooms,
        k_values=k_vals
    )
    all_adaptation_results.extend(res_b_to_a)

    # Format Results
    df_results = pd.DataFrame(all_adaptation_results)
    print("\n" + "=" * 65)
    print("Task 3 Few-Shot Adaptation Benchmark Results:")
    print(df_results.to_string(index=False))
    print("=" * 65)

    # Save deliverables
    res_csv = checkpoints_dir / "task3_few_shot_results.csv"
    df_results.to_csv(res_csv, index=False)

    cm_file = checkpoints_dir / "task3_confusion_matrices_k5.json"
    with open(cm_file, "w") as f:
        json.dump({"target_Room_B": cm_b_k5, "target_Room_A": cm_a_k5}, f, indent=2)

    print(f"\nSaved benchmark curve data to: {res_csv.resolve()}")
    print(f"Saved K=5 confusion matrices to: {cm_file.resolve()}")

if __name__ == "__main__":
    run_task3()