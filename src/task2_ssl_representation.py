import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Enforce deterministic behavior and enable cuDNN benchmark on GPU
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# 1. Self-Supervised Augmentation & Dataset
# ---------------------------------------------------------
class CSIAugmenter:
    """Domain-specific augmentations for WiFi CSI amplitude tensors."""
    def __init__(self, mask_ratio: float = 0.15, jitter_std: float = 0.05, scale_range=(0.85, 1.15)):
        self.mask_ratio = mask_ratio
        self.jitter_std = jitter_std
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Channels=256, Time=150)
        aug_x = x.clone()

        # 1. Subcarrier Masking (randomly zero out subcarriers)
        n_channels = aug_x.shape[0]
        mask_idx = torch.randperm(n_channels)[: int(n_channels * self.mask_ratio)]
        aug_x[mask_idx, :] = 0.0

        # 2. Additive Gaussian Jitter
        noise = torch.randn_like(aug_x) * self.jitter_std
        aug_x = aug_x + noise

        # 3. Random Amplitude Scaling
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        aug_x = aug_x * scale

        return aug_x

class UnlabeledCSIDataset(Dataset):
    """Loads development windows without activity or room labels for SSL."""
    def __init__(self, manifest_df: pd.DataFrame, augment: bool = True):
        self.records = manifest_df.reset_index(drop=True)
        self.augment = augment
        self.augmenter = CSIAugmenter()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path = self.records.iloc[idx]["file_path"]
        data = np.load(path).astype(np.float32)  # (150, 256)
        tensor_x = torch.from_numpy(data.T)      # (256, 150)

        if self.augment:
            view1 = self.augmenter(tensor_x)
            view2 = self.augmenter(tensor_x)
            return view1, view2
        return tensor_x

class LabeledCSIDataset(Dataset):
    """Loads windows with unified activity labels for Linear Probing."""
    def __init__(self, manifest_df: pd.DataFrame, label_to_idx: dict):
        self.records = manifest_df[manifest_df["activity"].isin(label_to_idx.keys())].reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        data = np.load(row["file_path"]).astype(np.float32)  # (150, 256)
        tensor_x = torch.from_numpy(data.T)                  # (256, 150)
        label = torch.tensor(self.label_to_idx[row["activity"]], dtype=torch.long)
        return tensor_x, label, row["room"], row["activity"]

# ---------------------------------------------------------
# 2. CSI Encoder & Contrastive Projection Head
# ---------------------------------------------------------
class CSIEncoderBackbone(nn.Module):
    """1D-CNN backbone mapping (256, 150) -> 128-dimensional latent vector."""
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

class SimCLRModel(nn.Module):
    """Backbone + Non-linear Projection Head for InfoNCE Pretraining."""
    def __init__(self, backbone: CSIEncoderBackbone, embedding_dim: int = 128, proj_dim: int = 64):
        super(SimCLRModel, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, proj_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, F.normalize(projections, dim=-1)

def info_nce_loss(z1, z2, temperature: float = 0.1):
    """Calculates InfoNCE (NT-Xent) contrastive loss between paired augmented views."""
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # (2*B, D)
    sim_matrix = torch.matmul(representations, representations.T) / temperature

    # Mask out self-contrast cases
    sim_i_j = torch.diag(sim_matrix, batch_size)
    sim_j_i = torch.diag(sim_matrix, -batch_size)
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).view(2 * batch_size, 1)

    mask = ~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z1.device)
    negative_samples = sim_matrix[mask].view(2 * batch_size, -1)

    logits = torch.cat([positive_samples, negative_samples], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels)

# ---------------------------------------------------------
# 3. Handcrafted Feature Extraction Baseline
# ---------------------------------------------------------
# 
def extract_handcrafted_features(npy_file_path: str) -> np.ndarray:
    """Extracts statistical & spectral summary features across subcarriers and time."""
    data = np.load(npy_file_path).astype(np.float32)  # (150, 256)
    
    # Temporal statistics per subcarrier (1D arrays of length 256)
    t_mean = np.mean(data, axis=0)
    t_std = np.std(data, axis=0)
    t_max = np.max(data, axis=0)
    t_min = np.min(data, axis=0)
    t_energy = np.mean(data**2, axis=0)

    # Scalar summary statistics wrapped explicitly as 1D float array
    global_scalars = np.array([
        float(np.mean(data)),
        float(np.std(data)),
        float(np.median(data)),
        float(np.percentile(data, 25)),
        float(np.percentile(data, 75)),
        float(np.mean(t_mean)),
        float(np.std(t_mean)),
        float(np.mean(t_std)),
        float(np.std(t_std)),
        float(np.mean(t_energy)),
        float(np.std(t_energy)),
        float(np.mean(np.diff(data, axis=0)**2))  # Temporal velocity proxy
    ], dtype=np.float32)

    # Concatenate all 1D arrays into a single flat vector
    features = np.concatenate([
        global_scalars,
        t_mean,
        t_std,
        t_energy
    ], axis=0)
    
    return features.astype(np.float32)
# ---------------------------------------------------------
# 4. Feature Extraction & Linear Evaluation
# ---------------------------------------------------------
def extract_embeddings(model, dataloader, device):
    """Passes data through frozen encoder on GPU and collects feature vectors."""
    model.eval()
    all_features, all_labels, all_rooms, all_activities = [], [], [], []

    with torch.no_grad():
        for inputs, labels, rooms, activities in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            feats = model(inputs)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_rooms.extend(rooms)
            all_activities.extend(activities)

    return np.vstack(all_features), np.array(all_labels), all_rooms, all_activities

# ---------------------------------------------------------
# 5. Main Execution Pipeline
# ---------------------------------------------------------
def run_task2():
    base_dir = Path("D:/Ub_Hack/hack_2")
    manifest_path = base_dir / "windows/windows_manifest.csv"
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Explicit GPU configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware accelerator: {device}")
    if device.type == "cuda":
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")

    df_manifest = pd.read_csv(manifest_path)
    df_dev = df_manifest[df_manifest["split"] == "development"].copy()
    df_val = df_manifest[df_manifest["split"] == "validation"].copy()

    # Unified activity vocabulary
    unique_activities = sorted(df_manifest["activity"].unique())
    label_to_idx = {act: idx for idx, act in enumerate(unique_activities)}
    print(f"\nActivity Classes ({len(unique_activities)}): {label_to_idx}")

    # =========================================================
    # Phase A: Self-Supervised InfoNCE Pretraining (Unlabeled Dev)
    # =========================================================
    print("\n" + "=" * 65)
    print("Phase A: Self-Supervised Contrastive Pretraining (SimCLR)")
    print(f"Unlabeled training set: {len(df_dev)} development windows (both rooms)")
    print("=" * 65)

    ssl_dataset = UnlabeledCSIDataset(df_dev, augment=True)
    ssl_loader = DataLoader(
        ssl_dataset, 
        batch_size=64, 
        shuffle=True, 
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )

    backbone = CSIEncoderBackbone(in_channels=256, embedding_dim=128).to(device)
    ssl_model = SimCLRModel(backbone, embedding_dim=128, proj_dim=64).to(device)
    optimizer = optim.AdamW(ssl_model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 20
    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss, steps = 0.0, 0
        for v1, v2 in ssl_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, z1 = ssl_model(v1)
            _, z2 = ssl_model(v2)
            loss = info_nce_loss(z1, z2, temperature=0.1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"SSL Epoch [{epoch:02d}/{epochs:02d}] | InfoNCE Loss: {total_loss / steps:.4f}")

    # Save frozen SSL encoder checkpoint (crucial for Task 3)
    ssl_checkpoint_path = checkpoints_dir / "ssl_encoder.pth"
    torch.save(backbone.state_dict(), ssl_checkpoint_path)
    print(f"\nPretrained SSL Backbone saved to: {ssl_checkpoint_path.resolve()}")

    # =========================================================
    # Phase B: Linear Probing Evaluation & Baseline Comparison
    # =========================================================
    print("\n" + "=" * 65)
    print("Phase B: Linear Probing Evaluation on Validation Set")
    print("=" * 65)

    # 1. Trained SSL Representation
    labeled_dev_dataset = LabeledCSIDataset(df_dev, label_to_idx)
    labeled_val_dataset = LabeledCSIDataset(df_val, label_to_idx)

    loader_kwargs = {"batch_size": 64, "shuffle": False, "pin_memory": (device.type == "cuda")}
    dev_loader = DataLoader(labeled_dev_dataset, **loader_kwargs)
    val_loader = DataLoader(labeled_val_dataset, **loader_kwargs)

    X_train_ssl, y_train, _, _ = extract_embeddings(backbone, dev_loader, device)
    X_val_ssl, y_val, val_rooms, val_acts = extract_embeddings(backbone, val_loader, device)

    clf_ssl = LogisticRegression(max_iter=1000, C=1.0)
    clf_ssl.fit(X_train_ssl, y_train)
    y_pred_ssl = clf_ssl.predict(X_val_ssl)

    ssl_f1 = f1_score(y_val, y_pred_ssl, average="macro")
    ssl_bal_acc = balanced_accuracy_score(y_val, y_pred_ssl)

    # 2. Random Weights Baseline (Untrained Architecture)
    random_backbone = CSIEncoderBackbone(in_channels=256, embedding_dim=128).to(device)
    X_train_rand, _, _, _ = extract_embeddings(random_backbone, dev_loader, device)
    X_val_rand, _, _, _ = extract_embeddings(random_backbone, val_loader, device)

    clf_rand = LogisticRegression(max_iter=1000, C=1.0)
    clf_rand.fit(X_train_rand, y_train)
    y_pred_rand = clf_rand.predict(X_val_rand)

    rand_f1 = f1_score(y_val, y_pred_rand, average="macro")
    rand_bal_acc = balanced_accuracy_score(y_val, y_pred_rand)

    # 3. Handcrafted Feature Baseline
    X_train_hc = np.array([extract_handcrafted_features(p) for p in df_dev["file_path"]])
    X_val_hc = np.array([extract_handcrafted_features(p) for p in df_val["file_path"]])
    y_train_hc = np.array([label_to_idx[a] for a in df_dev["activity"]])
    y_val_hc = np.array([label_to_idx[a] for a in df_val["activity"]])

    scaler = StandardScaler()
    X_train_hc_s = scaler.fit_transform(X_train_hc)
    X_val_hc_s = scaler.transform(X_val_hc)

    clf_hc = LogisticRegression(max_iter=1000, C=1.0)
    clf_hc.fit(X_train_hc_s, y_train_hc)
    y_pred_hc = clf_hc.predict(X_val_hc_s)

    hc_f1 = f1_score(y_val_hc, y_pred_hc, average="macro")
    hc_bal_acc = balanced_accuracy_score(y_val_hc, y_pred_hc)

    # Format & print comparison table
    results_summary = [
        {"Model Representation": "Handcrafted Features", "Validation Macro-F1": round(hc_f1, 4), "Balanced Accuracy (%)": round(hc_bal_acc * 100, 2)},
        {"Model Representation": "Random Weights (Frozen)", "Validation Macro-F1": round(rand_f1, 4), "Balanced Accuracy (%)": round(rand_bal_acc * 100, 2)},
        {"Model Representation": "Pretrained SSL Encoder (Frozen)", "Validation Macro-F1": round(ssl_f1, 4), "Balanced Accuracy (%)": round(ssl_bal_acc * 100, 2)}
    ]
    results_df = pd.DataFrame(results_summary)
    results_csv = checkpoints_dir / "task2_evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)

    print("\n" + "=" * 65)
    print("Task 2 Final Deliverable Comparison:")
    print(results_df.to_string(index=False))
    print("=" * 65)

    # =========================================================
    # Phase C: 2-D t-SNE Embedding Diagnosis Data
    # =========================================================
    print("\nComputing 2D t-SNE latent coordinates for report diagnostic figures...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_2d = tsne.fit_transform(X_val_ssl)

    tsne_df = pd.DataFrame({
        "tsne_dim1": tsne_2d[:, 0],
        "tsne_dim2": tsne_2d[:, 1],
        "room": val_rooms,
        "activity": val_acts
    })
    tsne_csv = checkpoints_dir / "task2_tsne_coordinates.csv"
    tsne_df.to_csv(tsne_csv, index=False)
    print(f"t-SNE coordinates saved for plotting: {tsne_csv.resolve()}")

if __name__ == "__main__":
    run_task2()