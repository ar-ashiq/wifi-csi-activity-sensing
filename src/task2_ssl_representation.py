from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import seaborn as sns

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from paths import CHECKPOINTS_DIR, PROJECT_ROOT, WINDOWS_DIR


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.1

EMBEDDING_DIM = 128
PROJECTION_DIM = 64

RANDOM_SEED = 42


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------
# 2. Device Configuration
# ---------------------------------------------------------

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Hardware accelerator: {device}")

    if device.type == "cuda":
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")

    return device


# ---------------------------------------------------------
# 3. Path Resolution
# ---------------------------------------------------------

def resolve_file_path(path):

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


# ---------------------------------------------------------
# 4. Self-Supervised Augmentation & Dataset
# ---------------------------------------------------------

class CSIAugmenter:
    """Domain-specific augmentations for WiFi CSI amplitude tensors."""

    def __init__(
        self,
        mask_ratio: float = 0.15,
        jitter_std: float = 0.05,
        scale_range=(0.85, 1.15)
    ):

        self.mask_ratio = mask_ratio
        self.jitter_std = jitter_std
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        # Input shape: (Channels=256, Time=150)

        aug_x = x.clone()

        # 1. Subcarrier Masking (randomly zero out subcarriers)

        n_channels = aug_x.shape[0]

        mask_idx = torch.randperm(n_channels)[
            :int(n_channels * self.mask_ratio)
        ]

        aug_x[mask_idx, :] = 0.0

        # 2. Additive Gaussian Jitter

        noise = torch.randn_like(aug_x) * self.jitter_std

        aug_x = aug_x + noise

        # 3. Random Amplitude Scaling

        scale = torch.empty(1).uniform_(
            *self.scale_range
        ).item()

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

        path = resolve_file_path(
            self.records.iloc[idx]["file_path"]
        )

        data = np.load(path).astype(np.float32)

        # Original shape: (150, 256)
        # CNN input shape: (256, 150)

        tensor_x = torch.from_numpy(data.T)

        if self.augment:

            view1 = self.augmenter(tensor_x)

            view2 = self.augmenter(tensor_x)

            return view1, view2

        return tensor_x


class LabeledCSIDataset(Dataset):
    """Loads windows with unified activity labels for Linear Probing."""

    def __init__(self, manifest_df: pd.DataFrame, label_to_idx: dict):

        self.records = manifest_df[
            manifest_df["activity"].isin(label_to_idx.keys())
        ].reset_index(drop=True)

        self.label_to_idx = label_to_idx

    def __len__(self):

        return len(self.records)

    def __getitem__(self, idx):

        row = self.records.iloc[idx]

        file_path = resolve_file_path(row["file_path"])

        data = np.load(file_path).astype(np.float32)

        # Original shape: (150, 256)
        # CNN input shape: (256, 150)

        tensor_x = torch.from_numpy(data.T)

        label = torch.tensor(
            self.label_to_idx[row["activity"]],
            dtype=torch.long
        )

        return tensor_x, label, row["room"], row["activity"]


# ---------------------------------------------------------
# 5. CSI Encoder & Contrastive Projection Head
# ---------------------------------------------------------

class CSIEncoderBackbone(nn.Module):
    """1D-CNN backbone mapping (256, 150) -> 128-dimensional latent vector."""

    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 128
    ):

        super(CSIEncoderBackbone, self).__init__()

        self.conv_blocks = nn.Sequential(

            nn.Conv1d(
                in_channels,
                128,
                kernel_size=7,
                stride=1,
                padding=3
            ),

            nn.BatchNorm1d(128),

            nn.GELU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                128,
                128,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.GELU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                128,
                embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm1d(embedding_dim),

            nn.GELU(),

            nn.AdaptiveAvgPool1d(1)

        )

    def forward(self, x):

        feat = self.conv_blocks(x)

        return feat.view(feat.size(0), -1)


class SimCLRModel(nn.Module):
    """Backbone + Non-linear Projection Head for InfoNCE Pretraining."""

    def __init__(
        self,
        backbone: CSIEncoderBackbone,
        embedding_dim: int = 128,
        proj_dim: int = 64
    ):

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


# ---------------------------------------------------------
# 6. Corrected InfoNCE Contrastive Loss
# ---------------------------------------------------------

def info_nce_loss(z1, z2, temperature: float = 0.1):
    """
    Calculates NT-Xent / InfoNCE contrastive loss.

    Each anchor has:
    - One positive sample
    - (2B - 2) negative samples

    Self-comparisons and positive duplicates are excluded
    from the negative samples.
    """

    batch_size = z1.size(0)

    if batch_size < 2:
        raise ValueError(
            "InfoNCE requires at least 2 original samples per batch."
        )

    representations = torch.cat(
        [z1, z2],
        dim=0
    )

    # Similarity matrix: (2B, 2B)

    sim_matrix = torch.matmul(
        representations,
        representations.T
    ) / temperature

    # Positive pairs:
    # z1[i] <-> z2[i]

    sim_i_j = torch.diag(
        sim_matrix,
        batch_size
    )

    sim_j_i = torch.diag(
        sim_matrix,
        -batch_size
    )

    positive_samples = torch.cat(
        [sim_i_j, sim_j_i],
        dim=0
    ).view(2 * batch_size, 1)

    # Identify positive-pair positions

    positive_mask = torch.zeros_like(
        sim_matrix,
        dtype=torch.bool
    )

    idx = torch.arange(
        batch_size,
        device=z1.device
    )

    positive_mask[idx, idx + batch_size] = True

    positive_mask[idx + batch_size, idx] = True

    # Remove self-comparisons

    self_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=z1.device
    )

    # Remove both self-comparisons and positive pairs

    negative_mask = ~(
        positive_mask | self_mask
    )

    negative_samples = sim_matrix[
        negative_mask
    ].view(
        2 * batch_size,
        2 * batch_size - 2
    )

    # Positive sample is always the first logit

    logits = torch.cat(
        [positive_samples, negative_samples],
        dim=1
    )

    labels = torch.zeros(
        2 * batch_size,
        dtype=torch.long,
        device=z1.device
    )

    return F.cross_entropy(
        logits,
        labels
    )


# ---------------------------------------------------------
# 7. Contrastive Validation Loss
# ---------------------------------------------------------

def evaluate_contrastive_loss(model, dataloader, device):

    model.eval()

    total_loss = 0.0

    steps = 0

    with torch.no_grad():

        for v1, v2 in dataloader:

            v1 = v1.to(device)

            v2 = v2.to(device)

            _, z1 = model(v1)

            _, z2 = model(v2)

            loss = info_nce_loss(
                z1,
                z2,
                temperature=TEMPERATURE
            )

            total_loss += loss.item()

            steps += 1

    if steps == 0:
        return float("nan")

    return total_loss / steps


# ---------------------------------------------------------
# 8. Handcrafted Feature Extraction Baseline
# ---------------------------------------------------------

def extract_handcrafted_features(npy_file_path: str) -> np.ndarray:
    """Extracts statistical & spectral summary features across subcarriers and time."""

    npy_file_path = resolve_file_path(npy_file_path)

    data = np.load(npy_file_path).astype(np.float32)

    # Temporal statistics per subcarrier (1D arrays of length 256)

    t_mean = np.mean(data, axis=0)

    t_std = np.std(data, axis=0)

    t_max = np.max(data, axis=0)

    t_min = np.min(data, axis=0)

    t_energy = np.mean(data**2, axis=0)

    # Scalar summary statistics

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

        float(np.mean(np.diff(data, axis=0)**2))

    ], dtype=np.float32)

    # Concatenate all features

    features = np.concatenate([

        global_scalars,

        t_mean,

        t_std,

        t_energy

    ], axis=0)

    return features.astype(np.float32)


# ---------------------------------------------------------
# 9. Feature Extraction & Linear Evaluation
# ---------------------------------------------------------

def extract_embeddings(model, dataloader, device):
    """Passes data through frozen encoder and collects feature vectors."""

    model.eval()

    all_features = []

    all_labels = []

    all_rooms = []

    all_activities = []

    with torch.no_grad():

        for inputs, labels, rooms, activities in dataloader:

            inputs = inputs.to(device)

            feats = model(inputs)

            all_features.append(
                feats.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            all_rooms.extend(rooms)

            all_activities.extend(activities)

    return (
        np.vstack(all_features),
        np.array(all_labels),
        all_rooms,
        all_activities
    )


# ---------------------------------------------------------
# 10. Plotting Functions
# ---------------------------------------------------------

def plot_training_loss(train_losses, val_losses, plots_dir):

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        marker="o",
        label="Training InfoNCE Loss"
    )

    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        marker="s",
        label="Validation InfoNCE Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("InfoNCE Loss")

    plt.title("SimCLR Contrastive Pretraining Loss")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        plots_dir / "01_ssl_training_loss.png",
        dpi=300
    )

    plt.close()


def plot_tsne(
    embeddings,
    labels,
    title,
    filename,
    plots_dir
):

    n_samples = len(embeddings)

    if n_samples < 3:

        print(
            f"Skipping {filename}: insufficient samples for t-SNE."
        )

        return None

    perplexity = min(
        30,
        max(1, (n_samples - 1) // 3)
    )

    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        perplexity=perplexity
    )

    tsne_2d = tsne.fit_transform(embeddings)

    labels = np.asarray(labels)

    plt.figure(figsize=(10, 7))

    unique_labels = sorted(
        pd.unique(labels),
        key=str
    )

    for label in unique_labels:

        mask = labels == label

        plt.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            label=str(label),
            alpha=0.7,
            s=25
        )

    plt.title(title)

    plt.xlabel("t-SNE Dimension 1")

    plt.ylabel("t-SNE Dimension 2")

    plt.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        plots_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return tsne_2d


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    plots_dir,
    model_name,
    filename
):
    """Saves a confusion matrix for one model representation."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    plt.figure(figsize=(8, 6), dpi=300)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        annot_kws={"size": 10}
    )

    plt.title(
        f"Confusion Matrix - {model_name}",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlabel("Predicted Label", fontsize=11)

    plt.ylabel("True Label", fontsize=11)

    plt.xticks(rotation=45, ha="right")

    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        plots_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def get_per_class_metrics(y_true, y_pred, class_names, model_name):
    """Returns per-class precision, recall, F1 and support for one model."""

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    rows = []

    for class_name in class_names:

        rows.append({
            "Model": model_name,
            "Activity": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1-score": report[class_name]["f1-score"],
            "Support": report[class_name]["support"]
        })

    return pd.DataFrame(rows)


def plot_per_class_performance(
    metrics_df,
    model_name,
    plots_dir,
    filename
):
    """Plots per-class precision, recall and F1 for one model."""

    model_df = metrics_df[
        metrics_df["Model"] == model_name
    ].set_index("Activity")[
        ["Precision", "Recall", "F1-score"]
    ]

    ax = model_df.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(f"Per-Class Performance - {model_name}")

    ax.set_xlabel("Activity")

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1)

    ax.set_xticklabels(
        model_df.index,
        rotation=35,
        ha="right"
    )

    ax.grid(axis="y", alpha=0.3)

    ax.legend(title="Metric")

    plt.tight_layout()

    plt.savefig(
        plots_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_per_class_f1_comparison(metrics_df, plots_dir, filename):
    """Compares per-class F1 across all three representations."""

    pivot_df = metrics_df.pivot(
        index="Activity",
        columns="Model",
        values="F1-score"
    )

    # Keep a consistent model order

    model_order = [
        "Handcrafted Features",
        "Random Weights (Frozen)",
        "Pretrained SSL Encoder (Frozen)"
    ]

    pivot_df = pivot_df[model_order]

    ax = pivot_df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_title("Per-Class F1-score Comparison Across Representations")

    ax.set_xlabel("Activity")

    ax.set_ylabel("F1-score")

    ax.set_ylim(0, 1)

    ax.set_xticklabels(
        pivot_df.index,
        rotation=35,
        ha="right"
    )

    ax.grid(axis="y", alpha=0.3)

    ax.legend(title="Model")

    plt.tight_layout()

    plt.savefig(
        plots_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_model_comparison(results_df, plots_dir, filename):

    plot_df = results_df.set_index(
        "Model Representation"
    )[
        [
            "Validation Macro-F1",
            "Balanced Accuracy (%)"
        ]
    ].copy()

    # Convert Macro-F1 into percentage for comparable axes

    plot_df["Validation Macro-F1"] *= 100

    ax = plot_df.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title(
        "Representation Learning Performance Comparison"
    )

    ax.set_ylabel("Score (%)")

    ax.set_xlabel("Model Representation")

    ax.set_ylim(0, 100)

    ax.set_xticklabels(
        plot_df.index,
        rotation=20,
        ha="right"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend(
        ["Macro-F1", "Balanced Accuracy"]
    )

    plt.tight_layout()

    plt.savefig(
        plots_dir / filename,
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 11. Positive vs Negative Similarity Distribution
# ---------------------------------------------------------

def plot_similarity_distribution(
    model,
    dataloader,
    device,
    plots_dir,
    filename
):
    """
    Measures cosine similarity between:
    - Positive pairs: two views of the same CSI window.
    - Negative pairs: views of different CSI windows.
    """

    model.eval()

    positive_similarities = []

    negative_similarities = []

    with torch.no_grad():

        for v1, v2 in dataloader:

            v1 = v1.to(device)

            v2 = v2.to(device)

            _, z1 = model(v1)

            _, z2 = model(v2)

            batch_size = z1.size(0)

            if batch_size < 2:
                continue

            representations = torch.cat(
                [z1, z2],
                dim=0
            )

            sim_matrix = torch.matmul(
                representations,
                representations.T
            )

            # Positive pair similarities

            positive = torch.cat([

                torch.diag(sim_matrix, batch_size),

                torch.diag(sim_matrix, -batch_size)

            ])

            positive_similarities.extend(
                positive.cpu().numpy()
            )

            # Exclude self-comparisons

            positive_mask = torch.zeros_like(
                sim_matrix,
                dtype=torch.bool
            )

            idx = torch.arange(
                batch_size,
                device=device
            )

            positive_mask[idx, idx + batch_size] = True

            positive_mask[idx + batch_size, idx] = True

            self_mask = torch.eye(
                2 * batch_size,
                dtype=torch.bool,
                device=device
            )

            negative_mask = ~(
                positive_mask | self_mask
            )

            negatives = sim_matrix[
                negative_mask
            ]

            negative_similarities.extend(
                negatives.cpu().numpy()
            )

    if not positive_similarities or not negative_similarities:

        print("Skipping similarity distribution: insufficient samples.")

        return

    plt.figure(figsize=(9, 5))

    plt.hist(
        positive_similarities,
        bins=30,
        alpha=0.6,
        density=True,
        label="Positive pairs"
    )

    plt.hist(
        negative_similarities,
        bins=30,
        alpha=0.6,
        density=True,
        label="Negative pairs"
    )

    plt.xlabel("Cosine Similarity")

    plt.ylabel("Density")

    plt.title(
        "Positive vs Negative CSI Representation Similarity"
    )

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        plots_dir / filename,
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 12. Main Execution Pipeline
# ---------------------------------------------------------

def run_task2():

    manifest_path = WINDOWS_DIR / "windows_manifest.csv"

    checkpoints_dir = CHECKPOINTS_DIR

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plots_dir = checkpoints_dir / "plots_task2"

    plots_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    # -----------------------------------------------------
    # Load Dataset Manifest
    # -----------------------------------------------------

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df_manifest = pd.read_csv(manifest_path)

    required_columns = {
        "file_path",
        "split",
        "activity",
        "room"
    }

    missing_columns = required_columns - set(
        df_manifest.columns
    )

    if missing_columns:

        raise ValueError(
            f"Missing manifest columns: {missing_columns}"
        )

    df_dev = df_manifest[
        df_manifest["split"] == "development"
    ].copy()

    df_val = df_manifest[
        df_manifest["split"] == "validation"
    ].copy()

    if len(df_dev) < 2:

        raise ValueError(
            "At least two development windows are required for InfoNCE."
        )

    if len(df_val) < 1:

        raise ValueError(
            "Validation dataset is empty."
        )

    # Unified activity vocabulary

    unique_activities = sorted(
        df_manifest["activity"].dropna().unique()
    )

    label_to_idx = {
        act: idx
        for idx, act in enumerate(unique_activities)
    }

    class_names = unique_activities

    print(
        f"\nActivity Classes ({len(unique_activities)}): "
        f"{label_to_idx}"
    )

    print(f"Development windows: {len(df_dev)}")

    print(f"Validation windows: {len(df_val)}")

    # -----------------------------------------------------
    # Phase A: Self-Supervised InfoNCE Pretraining
    # -----------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "Phase A: Self-Supervised Contrastive Pretraining (SimCLR)"
    )

    print(
        f"Unlabeled training set: {len(df_dev)} development windows"
    )

    print("=" * 65)

    ssl_dataset = UnlabeledCSIDataset(
        df_dev,
        augment=True
    )

    val_ssl_dataset = UnlabeledCSIDataset(
        df_val,
        augment=True
    )

    train_batch_size = min(
        BATCH_SIZE,
        len(ssl_dataset)
    )

    val_batch_size = min(
        BATCH_SIZE,
        len(val_ssl_dataset)
    )

    ssl_loader = DataLoader(

        ssl_dataset,

        batch_size=train_batch_size,

        shuffle=True,

        pin_memory=(device.type == "cuda"),

        drop_last=(len(ssl_dataset) >= BATCH_SIZE)

    )

    val_ssl_loader = DataLoader(

        val_ssl_dataset,

        batch_size=val_batch_size,

        shuffle=False,

        pin_memory=(device.type == "cuda"),

        drop_last=(len(val_ssl_dataset) >= BATCH_SIZE)

    )

    backbone = CSIEncoderBackbone(
        in_channels=256,
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    ssl_model = SimCLRModel(
        backbone,
        embedding_dim=EMBEDDING_DIM,
        proj_dim=PROJECTION_DIM
    ).to(device)

    optimizer = optim.AdamW(
        ssl_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_losses = []

    val_losses = []

    for epoch in range(1, EPOCHS + 1):

        ssl_model.train()

        total_loss = 0.0

        steps = 0

        for v1, v2 in ssl_loader:

            v1 = v1.to(device)

            v2 = v2.to(device)

            optimizer.zero_grad()

            _, z1 = ssl_model(v1)

            _, z2 = ssl_model(v2)

            loss = info_nce_loss(
                z1,
                z2,
                temperature=TEMPERATURE
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            steps += 1

        if steps == 0:

            raise RuntimeError(
                "No training batches were generated."
            )

        epoch_loss = total_loss / steps

        train_losses.append(epoch_loss)

        # Evaluate contrastive objective on held-out
        # validation windows without updating model weights.

        val_loss = evaluate_contrastive_loss(
            ssl_model,
            val_ssl_loader,
            device
        )

        val_losses.append(val_loss)

        print(
            f"SSL Epoch [{epoch:02d}/{EPOCHS:02d}] "
            f"| Train Loss: {epoch_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}"
        )

    # -----------------------------------------------------
    # Save Pretrained Encoder
    # -----------------------------------------------------

    ssl_checkpoint_path = (
        checkpoints_dir / "ssl_encoder.pth"
    )

    torch.save(
        backbone.state_dict(),
        ssl_checkpoint_path
    )

    print(
        f"\nPretrained SSL Backbone saved to: "
        f"{ssl_checkpoint_path.resolve()}"
    )

    # Save loss history

    loss_df = pd.DataFrame({

        "epoch": range(1, EPOCHS + 1),

        "training_loss": train_losses,

        "validation_loss": val_losses

    })

    loss_df.to_csv(
        checkpoints_dir / "ssl_loss_history.csv",
        index=False
    )

    plot_training_loss(
        train_losses,
        val_losses,
        plots_dir
    )

    # -----------------------------------------------------
    # Phase B: Linear Probing Evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "Phase B: Linear Probing Evaluation on Validation Set"
    )

    print("=" * 65)

    labeled_dev_dataset = LabeledCSIDataset(
        df_dev,
        label_to_idx
    )

    labeled_val_dataset = LabeledCSIDataset(
        df_val,
        label_to_idx
    )

    if len(labeled_dev_dataset) == 0 or len(labeled_val_dataset) == 0:

        raise ValueError(
            "Labeled development or validation dataset is empty."
        )

    loader_kwargs = {

        "batch_size": BATCH_SIZE,

        "shuffle": False,

        "pin_memory": (device.type == "cuda")

    }

    dev_loader = DataLoader(
        labeled_dev_dataset,
        **loader_kwargs
    )

    val_loader = DataLoader(
        labeled_val_dataset,
        **loader_kwargs
    )

    # -----------------------------------------------------
    # 1. Pretrained SSL Representation
    # -----------------------------------------------------

    X_train_ssl, y_train, _, _ = extract_embeddings(
        backbone,
        dev_loader,
        device
    )

    X_val_ssl, y_val, val_rooms, val_acts = extract_embeddings(
        backbone,
        val_loader,
        device
    )

    clf_ssl = LogisticRegression(
        max_iter=1000,
        C=1.0
    )

    clf_ssl.fit(
        X_train_ssl,
        y_train
    )

    y_pred_ssl = clf_ssl.predict(
        X_val_ssl
    )

    ssl_f1 = f1_score(
        y_val,
        y_pred_ssl,
        average="macro",
        zero_division=0
    )

    ssl_bal_acc = balanced_accuracy_score(
        y_val,
        y_pred_ssl
    )

    # -----------------------------------------------------
    # 2. Random Weights Baseline
    # -----------------------------------------------------

    random_backbone = CSIEncoderBackbone(
        in_channels=256,
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    X_train_rand, _, _, _ = extract_embeddings(
        random_backbone,
        dev_loader,
        device
    )

    X_val_rand, _, _, _ = extract_embeddings(
        random_backbone,
        val_loader,
        device
    )

    clf_rand = LogisticRegression(
        max_iter=1000,
        C=1.0
    )

    clf_rand.fit(
        X_train_rand,
        y_train
    )

    y_pred_rand = clf_rand.predict(
        X_val_rand
    )

    rand_f1 = f1_score(
        y_val,
        y_pred_rand,
        average="macro",
        zero_division=0
    )

    rand_bal_acc = balanced_accuracy_score(
        y_val,
        y_pred_rand
    )

    # -----------------------------------------------------
    # 3. Handcrafted Feature Baseline
    # -----------------------------------------------------

    X_train_hc = np.array([

        extract_handcrafted_features(p)

        for p in df_dev["file_path"]

    ])

    X_val_hc = np.array([

        extract_handcrafted_features(p)

        for p in df_val["file_path"]

    ])

    y_train_hc = np.array([

        label_to_idx[a]

        for a in df_dev["activity"]

    ])

    y_val_hc = np.array([

        label_to_idx[a]

        for a in df_val["activity"]

    ])

    scaler = StandardScaler()

    X_train_hc_s = scaler.fit_transform(
        X_train_hc
    )

    X_val_hc_s = scaler.transform(
        X_val_hc
    )

    clf_hc = LogisticRegression(
        max_iter=1000,
        C=1.0
    )

    clf_hc.fit(
        X_train_hc_s,
        y_train_hc
    )

    y_pred_hc = clf_hc.predict(
        X_val_hc_s
    )

    hc_f1 = f1_score(
        y_val_hc,
        y_pred_hc,
        average="macro",
        zero_division=0
    )

    hc_bal_acc = balanced_accuracy_score(
        y_val_hc,
        y_pred_hc
    )

    # -----------------------------------------------------
    # Model Comparison
    # -----------------------------------------------------

    results_summary = [

        {

            "Model Representation": "Handcrafted Features",

            "Validation Macro-F1": round(hc_f1, 4),

            "Balanced Accuracy (%)": round(
                hc_bal_acc * 100, 2
            )

        },

        {

            "Model Representation": "Random Weights (Frozen)",

            "Validation Macro-F1": round(rand_f1, 4),

            "Balanced Accuracy (%)": round(
                rand_bal_acc * 100, 2
            )

        },

        {

            "Model Representation": "Pretrained SSL Encoder (Frozen)",

            "Validation Macro-F1": round(ssl_f1, 4),

            "Balanced Accuracy (%)": round(
                ssl_bal_acc * 100, 2
            )

        }

    ]

    results_df = pd.DataFrame(
        results_summary
    )

    results_csv = (
        checkpoints_dir / "task2_evaluation_results.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False
    )

    print("\n" + "=" * 65)

    print(
        "Task 2 Final Comparisons:"
    )

    print(
        results_df.to_string(index=False)
    )

    print("=" * 65)

    # -----------------------------------------------------
    # Phase C: Classification Visualizations
    # -----------------------------------------------------

    print(
        "\nGenerating classification visualizations..."
    )

    # Confusion matrices for all three representations

    plot_confusion_matrix(
        y_val_hc,
        y_pred_hc,
        class_names,
        plots_dir,
        "Handcrafted Features",
        "04_confusion_matrix_handcrafted.png"
    )

    plot_confusion_matrix(
        y_val,
        y_pred_rand,
        class_names,
        plots_dir,
        "Random Weights (Frozen)",
        "05_confusion_matrix_random.png"
    )

    plot_confusion_matrix(
        y_val,
        y_pred_ssl,
        class_names,
        plots_dir,
        "Pretrained SSL Encoder (Frozen)",
        "06_confusion_matrix_ssl.png"
    )

    # Per-class precision / recall / F1 for every model

    per_class_df = pd.concat(
        [
            get_per_class_metrics(
                y_val_hc,
                y_pred_hc,
                class_names,
                "Handcrafted Features"
            ),
            get_per_class_metrics(
                y_val,
                y_pred_rand,
                class_names,
                "Random Weights (Frozen)"
            ),
            get_per_class_metrics(
                y_val,
                y_pred_ssl,
                class_names,
                "Pretrained SSL Encoder (Frozen)"
            )
        ],
        ignore_index=True
    )

    per_class_csv = (
        checkpoints_dir / "task2_per_class_metrics.csv"
    )

    per_class_df.to_csv(
        per_class_csv,
        index=False
    )

    plot_per_class_performance(
        per_class_df,
        "Handcrafted Features",
        plots_dir,
        "07_per_class_handcrafted.png"
    )

    plot_per_class_performance(
        per_class_df,
        "Random Weights (Frozen)",
        plots_dir,
        "08_per_class_random.png"
    )

    plot_per_class_performance(
        per_class_df,
        "Pretrained SSL Encoder (Frozen)",
        plots_dir,
        "09_per_class_ssl.png"
    )

    # Overall and per-class comparisons across the three models

    plot_model_comparison(
        results_df,
        plots_dir,
        "10_model_comparison.png"
    )

    plot_per_class_f1_comparison(
        per_class_df,
        plots_dir,
        "11_per_class_f1_comparison.png"
    )

    print(f"Per-class metrics saved to: {per_class_csv.resolve()}")

    # -----------------------------------------------------
    # Phase D: t-SNE Embedding Visualization
    # -----------------------------------------------------

    print(
        "\nComputing 2D t-SNE latent coordinates..."
    )

    tsne_2d = plot_tsne(

        X_val_ssl,

        val_acts,

        "t-SNE of SSL Embeddings - Activity",

        "02_tsne_by_activity.png",

        plots_dir

    )

    plot_tsne(

        X_val_ssl,

        val_rooms,

        "t-SNE of SSL Embeddings - Room",

        "03_tsne_by_room.png",

        plots_dir

    )

    if tsne_2d is not None:

        tsne_df = pd.DataFrame({

            "tsne_dim1": tsne_2d[:, 0],

            "tsne_dim2": tsne_2d[:, 1],

            "room": val_rooms,

            "activity": val_acts

        })

        tsne_csv = (
            checkpoints_dir / "task2_tsne_coordinates.csv"
        )

        tsne_df.to_csv(
            tsne_csv,
            index=False
        )

        print(
            f"t-SNE coordinates saved to: {tsne_csv.resolve()}"
        )

    # -----------------------------------------------------
    # Phase E: Positive vs Negative Similarity
    # -----------------------------------------------------

    print(
        "\nGenerating contrastive similarity distribution..."
    )

    plot_similarity_distribution(
        ssl_model,
        val_ssl_loader,
        device,
        plots_dir,
        "12_similarity_distribution.png"
    )

    # -----------------------------------------------------
    # Final Output
    # -----------------------------------------------------

    print("\n" + "=" * 65)

    print("TASK 2 EXECUTION COMPLETED")

    print("=" * 65)

    print(
        f"\nEncoder checkpoint: {ssl_checkpoint_path}"
    )

    print(
        f"Evaluation results: {results_csv}"
    )

    print(
        f"Per-class metrics: {per_class_csv}"
    )

    print(
        f"Plots directory: {plots_dir.resolve()}"
    )

    print("\nGenerated visualizations:")

    for plot_file in sorted(plots_dir.glob("*.png")):

        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    run_task2()
