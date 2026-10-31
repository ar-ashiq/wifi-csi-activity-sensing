import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from paths import CHECKPOINTS_DIR, PROJECT_ROOT, TASK1_ABLATION_CSV, WINDOWS_DIR

print(f"Check points: {CHECKPOINTS_DIR}")



# ---------------------------------------------------------
# 4. Visualization and Results Export
# ---------------------------------------------------------

def save_confusion_matrices(y_true, y_pred, plots_dir):
    class_names = ["Room_A", "Room_B"]

    # Standard confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    ax.set_title("Room Classification - Confusion Matrix")
    ax.set_xlabel("Predicted Room")
    ax.set_ylabel("Actual Room")

    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Normalized confusion matrix
    cm_normalized = confusion_matrix(
        y_true, y_pred, labels=[0, 1], normalize="true"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=class_names
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2%",
        colorbar=True
    )

    ax.set_title("Normalized Room Classification Matrix")
    ax.set_xlabel("Predicted Room")
    ax.set_ylabel("Actual Room")

    plt.tight_layout()
    plt.savefig(plots_dir / "normalized_confusion_matrix.png", dpi=300)
    plt.close()

    print("Confusion matrices saved.")


def save_classification_report(y_true, y_pred, plots_dir):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Room_A", "Room_B"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(plots_dir / "classification_report.csv")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Room_A", "Room_B"],
        zero_division=0
    ))


def save_training_plots(history, plots_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Training loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "training_loss.png", dpi=300)
    plt.close()

    # Training accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_acc"],
        marker="o",
        label="Training Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "training_accuracy.png", dpi=300)
    plt.close()

    # Validation accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["val_acc"],
        marker="o",
        label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_accuracy.png", dpi=300)
    plt.close()

    # Validation Macro F1
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["val_f1"],
        marker="o",
        label="Validation Macro-F1"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Validation Macro-F1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_macro_f1.png", dpi=300)
    plt.close()


def save_ablation_plot(ablation_df, plots_dir):
    plt.figure(figsize=(9, 5))

    plt.plot(
        ablation_df["window_length"],
        ablation_df["validation_accuracy"],
        marker="o",
        linewidth=2,
        label="Validation Accuracy"
    )

    plt.xlabel("Window Length (Observations)")
    plt.ylabel("Accuracy (%)")
    plt.title("Environment Fingerprint vs Window Length")
    plt.xticks(ablation_df["window_length"])
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "window_length_ablation.png", dpi=300)
    plt.close()


def save_class_distribution(df_dev, df_val, plots_dir):
    rooms = ["Room_A", "Room_B"]

    dev_counts = [
        (df_dev["room"] == room).sum() for room in rooms
    ]

    val_counts = [
        (df_val["room"] == room).sum() for room in rooms
    ]

    x = np.arange(len(rooms))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(x - width / 2, dev_counts, width, label="Development")
    plt.bar(x + width / 2, val_counts, width, label="Validation")

    plt.xlabel("Room")
    plt.ylabel("Number of Samples")
    plt.title("Dataset Class Distribution")
    plt.xticks(x, rooms)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# 1. Dataset Loader from Manifest
# ---------------------------------------------------------
class CSIRoomDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, window_len: int = 150):
        self.records = manifest_df.reset_index(drop=True)
        self.window_len = window_len
        # Label: Room_A -> 0, Room_B -> 1
        self.label_map = {"Room_A": 0, "Room_B": 1}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        file_path = row["file_path"]
        path = Path(file_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        data = np.load(path).astype(np.float32)  # Expected shape: (150, 256)

        # Slice window if evaluating window length ablation (T <= 150)
        if self.window_len < 150:
            start = (150 - self.window_len) // 2
            data = data[start : start + self.window_len, :]

        # Rearrange to shape: (Channels, Sequence_Length) = (256, T)
        # Treating subcarriers as channels for 1D Temporal Convolution
        tensor_data = torch.from_numpy(data.T)
        label = torch.tensor(self.label_map[row["room"]], dtype=torch.long)

        return tensor_data, label

# ---------------------------------------------------------
# 2. 1D-CNN Environment Fingerprint Architecture
# ---------------------------------------------------------
class RoomClassifier1D(nn.Module):
    def __init__(self, in_channels: int = 256):
        super(RoomClassifier1D, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Collapses temporal dimension to 1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits

# ---------------------------------------------------------
# 3. Training & Validation Procedures
# ---------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    return acc, f1, np.array(all_preds), np.array(all_targets)

# ---------------------------------------------------------
# 4. Main Execution Pipeline
# ---------------------------------------------------------

# ---------------------------------------------------------
# 5. Main Execution Pipeline
# ---------------------------------------------------------

def run_task1():

    manifest_path = WINDOWS_DIR / "windows_manifest.csv"

    checkpoints_dir = CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Separate directory for plots and evaluation results
    plots_dir = checkpoints_dir / "plots_task1"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_csv(manifest_path)

    # Strictly split by split column
    df_dev = df_manifest[
        df_manifest["split"] == "development"
    ].copy()

    df_val = df_manifest[
        df_manifest["split"] == "validation"
    ].copy()

    print("=" * 60)
    print("Task 1: Training Room Identification Classifier")
    print(f"Development samples: {len(df_dev)}")
    print(f"Validation samples : {len(df_val)}")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Running on device: {device}\n")

    # Dataset and dataloaders
    train_dataset = CSIRoomDataset(df_dev, window_len=150)
    val_dataset = CSIRoomDataset(df_val, window_len=150)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    # Model
    model = RoomClassifier1D(in_channels=256).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    best_val_acc = -1.0

    best_model_path = checkpoints_dir / "room_classifier.pth"

    # Store epoch-wise metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": []
    }

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\nStarting training...\n")

    for epoch in range(1, 16):

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc * 100)
        history["val_acc"].append(val_acc * 100)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch [{epoch:02d}/15] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # -----------------------------------------------------
    # Load Best Model
    # -----------------------------------------------------

    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )

    model.eval()

    print("\n" + "=" * 60)
    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )
    print(f"Model saved: {best_model_path.resolve()}")
    print("=" * 60)

    # -----------------------------------------------------
    # Final Validation Evaluation
    # -----------------------------------------------------

    val_acc, val_f1, predictions, targets = evaluate(
        model, val_loader, device
    )

    print("\nFinal Best Model Evaluation")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Macro-F1: {val_f1:.4f}")

    # Generate confusion matrices
    save_confusion_matrices(
        targets, predictions, plots_dir
    )

    # Generate classification report
    save_classification_report(
        targets, predictions, plots_dir
    )

    # Generate training plots
    save_training_plots(history, plots_dir)

    # Generate class distribution
    save_class_distribution(
        df_dev, df_val, plots_dir
    )

    # -----------------------------------------------------
    # Window Length Ablation
    # -----------------------------------------------------

    print(
        "\nRunning Environment Fingerprint Ablation "
        "(30, 60, 90, 120, 150)..."
    )

    ablation_results = []

    window_lengths = [30, 60, 90, 120, 150]

    for w_len in window_lengths:

        val_sub_dataset = CSIRoomDataset(
            df_val, window_len=w_len
        )

        val_sub_loader = DataLoader(
            val_sub_dataset,
            batch_size=32,
            shuffle=False
        )

        sub_acc, sub_f1, _, _ = evaluate(
            model, val_sub_loader, device
        )

        ablation_results.append({
            "window_length": w_len,
            "validation_accuracy": round(sub_acc * 100, 2),
            "macro_f1": round(sub_f1, 4)
        })

        print(
            f"Window Length = {w_len:3d} | "
            f"Accuracy = {sub_acc*100:.2f}% | "
            f"Macro-F1 = {sub_f1:.4f}"
        )

    # Save ablation CSV
    ablation_df = pd.DataFrame(ablation_results)

    ablation_df.to_csv(
        plots_dir / "ablation_results.csv",
        index=False
    )

    # Save existing report CSV location too
    ablation_df.to_csv(
        TASK1_ABLATION_CSV,
        index=False
    )

    # Plot ablation results
    save_ablation_plot(
        ablation_df, plots_dir
    )

    print("\n" + "=" * 60)
    print("TASK 1 COMPLETED")
    print(f"Checkpoint: {best_model_path.resolve()}")
    print(f"Plots and metrics: {plots_dir.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    run_task1()