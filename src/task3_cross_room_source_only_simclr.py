"""
Task 3: Cross-Room Few-Label Adaptation with source-only SimCLR pretraining.

Protocol:
- Train a separate SimCLR backbone for each transfer direction.
- SimCLR sees ONLY source-room DEVELOPMENT CSI windows.
- For each K, target support contains exactly K distinct target-exclusive
  original captures per exclusive class. No target shared-class labels are used.
- Target VALIDATION is used only for final feature extraction and evaluation.
- Evaluation is capture-level: all windows belonging to one original capture
  are averaged before prototype construction / prediction.

Required manifest columns:
    file_path, activity, room, split, capture_id
Optional:
    window_id

The existing manifest described in the baseline did not contain capture_id.
Add it from the original recording metadata before running this script.
Do NOT use window_id as capture_id unless it truly identifies the original
recording (rather than an individual window).

Expected CSI array per file: (150, 256), converted to (256, 150).
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------
# Project paths: preserve the layout used by the existing project.
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))
from paths import CHECKPOINTS_DIR, PROJECT_ROOT, WINDOWS_DIR


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MANIFEST_PATH = WINDOWS_DIR / "windows_manifest.csv"
OUTPUT_DIR = CHECKPOINTS_DIR / "task3_cross_room_outputs"

ROOM_A = "Room_A"
ROOM_B = "Room_B"

SHARED_CLASSES = ["SIT", "STAND", "WALK"]
ROOM_A_EXCLUSIVE = ["FALL", "SIT_DOWN", "STAND_UP"]
ROOM_B_EXCLUSIVE = ["EMPTY", "LYING"]

K_VALUES = [1, 5, 10, 20]
SUPPORT_SEEDS = [42, 43, 44, 45, 46]  # repeated support-set draws

# SimCLR settings
SSL_EPOCHS = 100
SSL_BATCH_SIZE = 128
SSL_LR = 1e-3
SSL_WEIGHT_DECAY = 1e-4
SSL_TEMPERATURE = 0.2
PROJECTION_DIM = 64

# Runtime
FEATURE_BATCH_SIZE = 128
NUM_WORKERS = 0
BASE_SEED = 42

# Expected input dimensions
CSI_CHANNELS = 256
CSI_TIME_STEPS = 150
EMBEDDING_DIM = 128


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic settings; benchmark is intentionally disabled.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class CSIWindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, with_metadata: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.with_metadata = with_metadata

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(str(row["file_path"]))
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(f"CSI window file not found: {path}")

        array = np.load(path).astype(np.float32)
        if array.shape != (CSI_TIME_STEPS, CSI_CHANNELS):
            raise ValueError(
                f"{path} has shape {array.shape}; expected "
                f"({CSI_TIME_STEPS}, {CSI_CHANNELS})."
            )

        x = torch.from_numpy(array.T.copy())  # (256, 150)

        if not self.with_metadata:
            return x

        return (
            x,
            str(row["activity"]),
            str(row["room"]),
            str(row["capture_id"]),
        )


# ---------------------------------------------------------------------
# Backbone: same CNN architecture as the supplied Task 3 baseline.
# ---------------------------------------------------------------------
class CSIEncoderBackbone(nn.Module):
    def __init__(self, in_channels=CSI_CHANNELS, embedding_dim=EMBEDDING_DIM):
        super().__init__()
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
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.conv_blocks(x).flatten(1)


class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CSIEncoderBackbone()
        self.projector = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.GELU(),
            nn.Linear(EMBEDDING_DIM, PROJECTION_DIM),
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projector(features)
        return features, projection


# ---------------------------------------------------------------------
# CSI augmentations for two SimCLR views.
# These are deliberately mild and do not use activity or room labels.
# ---------------------------------------------------------------------
def augment_csi(batch: torch.Tensor) -> torch.Tensor:
    x = batch.clone()
    batch_size, channels, time_steps = x.shape

    # Per-window amplitude scaling.
    scale = torch.empty(batch_size, 1, 1, device=x.device).uniform_(0.9, 1.1)
    x = x * scale

    # Small additive noise relative to each window's own scale.
    sample_std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    x = x + torch.randn_like(x) * (0.02 * sample_std)

    # Random short time mask (same masked time indices across channels).
    for i in range(batch_size):
        if random.random() < 0.5:
            mask_len = random.randint(1, max(1, int(0.10 * time_steps)))
            start = random.randint(0, time_steps - mask_len)
            x[i, :, start:start + mask_len] = 0.0

    # Randomly suppress a small number of channels.
    channel_mask = (
        torch.rand(batch_size, channels, 1, device=x.device) < 0.02
    )
    x = x.masked_fill(channel_mask, 0.0)

    return x


# ---------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------
def nt_xent_loss(z1, z2, temperature=SSL_TEMPERATURE):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.shape[0]
    representations = torch.cat([z1, z2], dim=0)  # (2B, D)

    logits = torch.matmul(representations, representations.T) / temperature
    logits.fill_diagonal_(-1e9)

    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z1.device),
        torch.arange(0, batch_size, device=z1.device),
    ])

    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------
# Source-only SimCLR training
# ---------------------------------------------------------------------
def train_source_simclr(source_dev: pd.DataFrame, source_room: str, device):
    ssl_dataset = CSIWindowDataset(source_dev, with_metadata=False)
    if len(ssl_dataset) < 2:
        raise ValueError(f"Not enough source windows for SimCLR in {source_room}.")

    loader = DataLoader(
        ssl_dataset,
        batch_size=SSL_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    if len(loader) == 0:
        raise ValueError(
            f"{source_room} has fewer than SSL_BATCH_SIZE={SSL_BATCH_SIZE} "
            "windows. Reduce SSL_BATCH_SIZE."
        )

    model = SimCLRModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SSL_LR, weight_decay=SSL_WEIGHT_DECAY
    )

    history = []
    print(f"\nTraining source-only SimCLR: {source_room}")
    print(f"Source development windows: {len(ssl_dataset)}")

    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()
        losses = []

        for x in loader:
            x = x.to(device, non_blocking=True)
            view1 = augment_csi(x)
            view2 = augment_csi(x)

            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "simclr_loss": epoch_loss})

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{source_room} | Epoch {epoch:03d}/{SSL_EPOCHS} "
                f"| NT-Xent loss: {epoch_loss:.5f}"
            )

    model.eval()
    return model.encoder, pd.DataFrame(history)


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
@torch.no_grad()
def extract_features(backbone, frame: pd.DataFrame, device):
    dataset = CSIWindowDataset(frame, with_metadata=True)
    loader = DataLoader(
        dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    backbone.eval()
    feature_parts = []
    activity_values, room_values, capture_values = [], [], []

    for x, activities, rooms, capture_ids in loader:
        x = x.to(device, non_blocking=True)
        embeddings = F.normalize(backbone(x), p=2, dim=1)
        feature_parts.append(embeddings.cpu().numpy())
        activity_values.extend(list(activities))
        room_values.extend(list(rooms))
        capture_values.extend(list(capture_ids))

    if not feature_parts:
        raise ValueError("Cannot extract features from an empty dataframe.")

    return (
        np.concatenate(feature_parts, axis=0),
        np.asarray(activity_values),
        np.asarray(room_values),
        np.asarray(capture_values),
    )


# ---------------------------------------------------------------------
# Capture-level aggregation and validation
# ---------------------------------------------------------------------
def capture_level_table(features, activities, rooms, capture_ids):
    rows = []
    frame = pd.DataFrame({
        "activity": activities,
        "room": rooms,
        "capture_id": capture_ids,
        "feature_index": np.arange(len(features)),
    })

    # A capture must map to one room and one activity.
    consistency = frame.groupby(["room", "capture_id"])["activity"].nunique()
    if (consistency > 1).any():
        bad = consistency[consistency > 1].index.tolist()[:5]
        raise ValueError(
            "A capture_id maps to multiple activities. Make capture_id unique "
            f"per original recording. Examples: {bad}"
        )

    for (room, activity, capture_id), group in frame.groupby(
        ["room", "activity", "capture_id"], sort=False
    ):
        vector = features[group["feature_index"].to_numpy()].mean(axis=0)
        vector = vector / (np.linalg.norm(vector) + 1e-12)
        rows.append({
            "room": room,
            "activity": activity,
            "capture_id": capture_id,
            "feature": vector.astype(np.float32),
            "num_windows": len(group),
        })

    return pd.DataFrame(rows)


def make_prototype(capture_vectors):
    prototype = np.mean(np.stack(capture_vectors), axis=0)
    return prototype / (np.linalg.norm(prototype) + 1e-12)


# ---------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, target_classes, shared_classes, exclusive_classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "overall_macro_f1": f1_score(
            y_true, y_pred, labels=target_classes, average="macro", zero_division=0
        ),
        "shared_macro_f1": f1_score(
            y_true, y_pred, labels=shared_classes, average="macro", zero_division=0
        ),
        "exclusive_macro_f1": f1_score(
            y_true, y_pred, labels=exclusive_classes, average="macro", zero_division=0
        ),
    }


def save_loss_plot(history, direction, output_dir):
    csv_path = output_dir / f"{direction}_simclr_loss.csv"
    png_path = output_dir / f"{direction}_simclr_loss.png"
    history.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["simclr_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent loss")
    plt.title(f"Source-only SimCLR training loss: {direction}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def save_metric_plots(summary, direction, output_dir):
    subset = summary[summary["direction"] == direction].sort_values("K")
    metric_specs = [
        ("overall_macro_f1", "Overall Macro-F1"),
        ("shared_macro_f1", "Shared-class Macro-F1"),
        ("exclusive_macro_f1", "Exclusive-class Macro-F1"),
        ("balanced_accuracy", "Balanced accuracy"),
    ]

    for metric, label in metric_specs:
        plt.figure(figsize=(7, 5))
        plt.errorbar(
            subset["K"],
            subset[f"{metric}_mean"],
            yerr=subset[f"{metric}_std"],
            marker="o",
            capsize=4,
        )
        plt.xscale("log", base=2)
        plt.xticks(K_VALUES, [str(k) for k in K_VALUES])
        plt.ylim(0, 1)
        plt.xlabel("K labelled target captures per exclusive activity")
        plt.ylabel(label)
        plt.title(f"{label} vs K: {direction}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{direction}_{metric}_vs_k.png", dpi=200)
        plt.close()


def plot_confusion_matrix_blue(cm, labels, title, path, normalize=False):
    """Blue-toned confusion matrix: raw counts or row-normalized (recall)."""
    cm = np.asarray(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(
            cm, row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )
        fmt, vmin, vmax = ".2f", 0.0, 1.0
        cbar_label = "Recall (row-normalized)"
    else:
        data = cm
        fmt, vmin, vmax = "d", None, None
        cbar_label = "Count"

    plt.figure(figsize=(8, 6), dpi=300)

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": cbar_label},
        annot_kws={"size": 10}
    )

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_plots(cm, labels, direction, k, output_dir):
    """Saves both the count and the normalized version for one K."""
    plot_confusion_matrix_blue(
        cm, labels,
        f"Confusion Matrix (counts, summed over seeds) - {direction}, K={k}",
        output_dir / f"confusion_matrix_K{k}_counts.png",
        normalize=False
    )
    plot_confusion_matrix_blue(
        cm, labels,
        f"Confusion Matrix (normalized) - {direction}, K={k}",
        output_dir / f"confusion_matrix_K{k}_normalized.png",
        normalize=True
    )


# ---------------------------------------------------------------------
# One transfer direction
# ---------------------------------------------------------------------
def run_direction(
    source_room,
    target_room,
    target_exclusive_classes,
    manifest_dev,
    manifest_val,
    device,
):
    direction = f"{source_room}_to_{target_room}"
    direction_dir = OUTPUT_DIR / direction
    direction_dir.mkdir(parents=True, exist_ok=True)

    # The backbone's entire SSL training set is source-room DEVELOPMENT only.
    source_dev = manifest_dev[manifest_dev["room"] == source_room].copy()
    if source_dev.empty:
        raise ValueError(f"No source development windows for {source_room}.")

    backbone, loss_history = train_source_simclr(source_dev, source_room, device)
    save_loss_plot(loss_history, direction, direction_dir)

    checkpoint_path = direction_dir / "source_only_simclr_backbone.pth"
    torch.save(backbone.state_dict(), checkpoint_path)
    print(f"Saved source-only backbone: {checkpoint_path}")

    # Source prototypes: use source development captures only.
    source_features, source_acts, source_rooms, source_capture_ids = extract_features(
        backbone, source_dev, device
    )
    source_capture_table = capture_level_table(
        source_features, source_acts, source_rooms, source_capture_ids
    )

    shared_prototypes = {}
    for activity in SHARED_CLASSES:
        vectors = source_capture_table.loc[
            (source_capture_table["room"] == source_room)
            & (source_capture_table["activity"] == activity),
            "feature",
        ].tolist()
        if not vectors:
            raise ValueError(
                f"No source-room captures for shared class {activity} in {source_room}."
            )
        shared_prototypes[activity] = make_prototype(vectors)

    # Target validation is used only for final inference/evaluation.
    target_classes = SHARED_CLASSES + target_exclusive_classes
    target_val = manifest_val[
        (manifest_val["room"] == target_room)
        & manifest_val["activity"].isin(target_classes)
    ].copy()

    if target_val.empty:
        raise ValueError(f"No target validation windows for {target_room}.")

    val_features, val_acts, val_rooms, val_capture_ids = extract_features(
        backbone, target_val, device
    )
    val_capture_table = capture_level_table(
        val_features, val_acts, val_rooms, val_capture_ids
    )
    val_capture_table = val_capture_table[
        val_capture_table["activity"].isin(target_classes)
    ].copy()

    missing_val = set(target_classes) - set(val_capture_table["activity"].unique())
    if missing_val:
        raise ValueError(
            f"Target validation set is missing required classes: {sorted(missing_val)}"
        )

    # Target development candidates are restricted to target-exclusive classes.
    # No target shared-class development examples are loaded for support.
    target_exclusive_dev = manifest_dev[
        (manifest_dev["room"] == target_room)
        & manifest_dev["activity"].isin(target_exclusive_classes)
    ].copy()

    # Count original captures, not windows.
    available = (
        target_exclusive_dev.groupby("activity")["capture_id"].nunique().to_dict()
    )
    for activity in target_exclusive_classes:
        if available.get(activity, 0) < max(K_VALUES):
            raise ValueError(
                f"{target_room}/{activity} has {available.get(activity, 0)} "
                f"distinct development captures; K=20 requires at least 20. "
                "Do not silently count windows as captures."
            )

    rows = []
    support_records = []
    confusion_by_k = defaultdict(lambda: np.zeros(
        (len(target_classes), len(target_classes)), dtype=np.int64
    ))
    class_reports = []

    for seed in SUPPORT_SEEDS:
        rng = np.random.default_rng(seed)

        for k in K_VALUES:
            selected_capture_ids = []
            for activity in target_exclusive_classes:
                ids = np.asarray(sorted(
                    target_exclusive_dev.loc[
                        target_exclusive_dev["activity"] == activity,
                        "capture_id"
                    ].astype(str).unique()
                ))
                chosen = rng.choice(ids, size=k, replace=False)
                selected_capture_ids.extend(
                    [(activity, str(capture_id)) for capture_id in chosen]
                )

            # Select ALL windows belonging to the chosen original captures.
            support_mask = np.zeros(len(target_exclusive_dev), dtype=bool)
            for activity, capture_id in selected_capture_ids:
                support_mask |= (
                    (target_exclusive_dev["activity"].to_numpy() == activity)
                    & (target_exclusive_dev["capture_id"].astype(str).to_numpy() == capture_id)
                )
            support_frame = target_exclusive_dev.loc[support_mask].copy()

            for activity, capture_id in selected_capture_ids:
                support_records.append({
                    "direction": direction,
                    "seed": seed,
                    "K": k,
                    "activity": activity,
                    "capture_id": capture_id,
                })

            support_features, support_acts, support_rooms, support_capture_ids = (
                extract_features(backbone, support_frame, device)
            )
            support_capture_table = capture_level_table(
                support_features, support_acts, support_rooms, support_capture_ids
            )

            # Target-exclusive prototypes from K capture-level support vectors.
            target_prototypes = {}
            for activity in target_exclusive_classes:
                vectors = support_capture_table.loc[
                    support_capture_table["activity"] == activity, "feature"
                ].tolist()
                if len(vectors) != k:
                    raise RuntimeError(
                        f"Expected {k} support captures for {activity}, got {len(vectors)}."
                    )
                target_prototypes[activity] = make_prototype(vectors)

            prototypes = {**shared_prototypes, **target_prototypes}
            prototype_matrix = np.stack(
                [prototypes[c] for c in target_classes], axis=0
            )

            X_val = np.stack(val_capture_table["feature"].to_list())
            y_true = val_capture_table["activity"].to_numpy()

            similarities = X_val @ prototype_matrix.T
            y_pred = np.asarray(target_classes)[np.argmax(similarities, axis=1)]

            metrics = evaluate_predictions(
                y_true, y_pred, target_classes, SHARED_CLASSES, target_exclusive_classes
            )
            rows.append({
                "direction": direction,
                "source_room": source_room,
                "target_room": target_room,
                "seed": seed,
                "K": k,
                "num_target_validation_captures": len(y_true),
                **metrics,
            })

            cm = confusion_matrix(y_true, y_pred, labels=target_classes)
            confusion_by_k[k] += cm

            report = classification_report(
                y_true,
                y_pred,
                labels=target_classes,
                output_dict=True,
                zero_division=0,
            )
            for label in target_classes:
                class_reports.append({
                    "direction": direction,
                    "seed": seed,
                    "K": k,
                    "class": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1_score": report[label]["f1-score"],
                    "support": report[label]["support"],
                })

            print(
                f"{direction} | seed={seed} | K={k} | "
                f"Macro-F1={metrics['overall_macro_f1']:.4f} | "
                f"Shared={metrics['shared_macro_f1']:.4f} | "
                f"Exclusive={metrics['exclusive_macro_f1']:.4f}"
            )

    # Save per-direction raw data and confusion matrices.
    pd.DataFrame(support_records).to_csv(
        direction_dir / "support_capture_selection.csv", index=False
    )
    pd.DataFrame(rows).to_csv(direction_dir / "per_seed_results.csv", index=False)
    pd.DataFrame(class_reports).to_csv(
        direction_dir / "per_class_metrics.csv", index=False
    )

    for k, cm in confusion_by_k.items():
        cm_df = pd.DataFrame(cm, index=target_classes, columns=target_classes)
        cm_df.to_csv(direction_dir / f"confusion_matrix_K{k}_counts.csv")

        save_confusion_plots(cm, target_classes, direction, k, direction_dir)

    return pd.DataFrame(rows), pd.DataFrame(class_reports)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    set_seed(BASE_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"file_path", "activity", "room", "split", "capture_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}. Add capture_id using original recording "
            "metadata; window_id is not automatically a valid substitute."
        )

    manifest["capture_id"] = manifest["capture_id"].astype(str)
    manifest["activity"] = manifest["activity"].astype(str)
    manifest["room"] = manifest["room"].astype(str)
    manifest["split"] = manifest["split"].astype(str).str.lower()

    development = manifest[manifest["split"] == "development"].copy()
    validation = manifest[manifest["split"] == "validation"].copy()

    if development.empty or validation.empty:
        raise ValueError("Manifest must contain development and validation rows.")

    # Verify that no original capture is split across development/validation.
    capture_split_counts = manifest.groupby(["room", "capture_id"])["split"].nunique()
    if (capture_split_counts > 1).any():
        examples = capture_split_counts[capture_split_counts > 1].index.tolist()[:10]
        raise ValueError(
            "Original captures appear in multiple splits. This violates the "
            f"capture-level split rule. Examples: {examples}"
        )

    # Verify labels/rooms are as expected.
    expected = {
        ROOM_A: set(SHARED_CLASSES + ROOM_A_EXCLUSIVE),
        ROOM_B: set(SHARED_CLASSES + ROOM_B_EXCLUSIVE),
    }
    for room, classes in expected.items():
        observed = set(manifest.loc[manifest["room"] == room, "activity"].unique())
        unexpected = observed - classes
        if unexpected:
            print(f"Warning: unexpected labels in {room}: {sorted(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Development windows: {len(development)}")
    print(f"Validation windows: {len(validation)}")
    print(f"Output directory: {OUTPUT_DIR}")

    all_results = []
    all_class_reports = []

    # A -> B: source-only backbone trained only on Room A development.
    results_ab, reports_ab = run_direction(
        source_room=ROOM_A,
        target_room=ROOM_B,
        target_exclusive_classes=ROOM_B_EXCLUSIVE,
        manifest_dev=development,
        manifest_val=validation,
        device=device,
    )
    all_results.append(results_ab)
    all_class_reports.append(reports_ab)

    # B -> A: a separately trained source-only backbone trained only on Room B.
    results_ba, reports_ba = run_direction(
        source_room=ROOM_B,
        target_room=ROOM_A,
        target_exclusive_classes=ROOM_A_EXCLUSIVE,
        manifest_dev=development,
        manifest_val=validation,
        device=device,
    )
    all_results.append(results_ba)
    all_class_reports.append(reports_ba)

    raw_results = pd.concat(all_results, ignore_index=True)
    class_reports = pd.concat(all_class_reports, ignore_index=True)

    raw_results.to_csv(OUTPUT_DIR / "all_directions_per_seed_results.csv", index=False)
    class_reports.to_csv(OUTPUT_DIR / "all_directions_per_class_metrics.csv", index=False)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "overall_macro_f1",
        "shared_macro_f1",
        "exclusive_macro_f1",
    ]
    summary = (
        raw_results.groupby(["direction", "K"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(OUTPUT_DIR / "mean_std_results_by_K.csv", index=False)

    save_metric_plots(summary, f"{ROOM_A}_to_{ROOM_B}", OUTPUT_DIR)
    save_metric_plots(summary, f"{ROOM_B}_to_{ROOM_A}", OUTPUT_DIR)

    print("\n================ TASK 3 SUMMARY ================")
    print(summary.to_string(index=False))
    print("\nSaved outputs under:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
