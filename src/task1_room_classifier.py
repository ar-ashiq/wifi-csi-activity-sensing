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

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from paths import CHECKPOINTS_DIR, PROJECT_ROOT, WINDOWS_DIR

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
def run_task1():
    manifest_path = WINDOWS_DIR / "windows_manifest.csv"
    checkpoints_dir = CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_csv(manifest_path)
    
    # Strictly split by 'split' column
    df_dev = df_manifest[df_manifest["split"] == "development"].copy()
    df_val = df_manifest[df_manifest["split"] == "validation"].copy()

    print("=" * 60)
    print(f"Task 1: Training Room Identification Classifier")
    print(f"Development samples : {len(df_dev)} (Room A: {(df_dev['room']=='Room_A').sum()}, Room B: {(df_dev['room']=='Room_B').sum()})")
    print(f"Validation samples  : {len(df_val)} (Room A: {(df_val['room']=='Room_A').sum()}, Room B: {(df_val['room']=='Room_B').sum()})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")

    # Dataloaders for official 150-length window
    train_dataset = CSIRoomDataset(df_dev, window_len=150)
    val_dataset = CSIRoomDataset(df_val, window_len=150)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = RoomClassifier1D(in_channels=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    best_model_path = checkpoints_dir / "room_classifier.pth"

    # Train for 15 epochs (convergence is fast on room fingerprinting)
    for epoch in range(1, 16):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch:02d}/15] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("\n" + "=" * 60)
    print(f"Best Headline Validation Accuracy (150 observations): {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {best_model_path.resolve()}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 5. Required Deliverable: Environment Fingerprint vs Window Length
    # ---------------------------------------------------------
    print("\nRunning Environment Fingerprint Ablation (Window Lengths: 30, 60, 90, 120, 150)...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    ablation_results = []
    window_lengths = [30, 60, 90, 120, 150]

    for w_len in window_lengths:
        val_sub_dataset = CSIRoomDataset(df_val, window_len=w_len)
        val_sub_loader = DataLoader(val_sub_dataset, batch_size=32, shuffle=False)
        sub_acc, sub_f1, _, _ = evaluate(model, val_sub_loader, device)
        ablation_results.append({
            "window_length": w_len,
            "validation_accuracy": round(sub_acc * 100, 2),
            "macro_f1": round(sub_f1, 4)
        })
        print(f"Window Length = {w_len:3d} | Accuracy = {sub_acc * 100:.2f}% | Macro-F1 = {sub_f1:.4f}")

    # Export results for plotting the required curve in the technical report
    ablation_df = pd.DataFrame(ablation_results)
    ablation_csv = base_dir / "checkpoints" / "task1_window_ablation.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"\nAblation metrics saved for report figure: {ablation_csv.resolve()}")

if __name__ == "__main__":
    run_task1()