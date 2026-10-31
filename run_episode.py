import os
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))


from paths import CHECKPOINTS_DIR, PROJECT_ROOT, WINDOWS_DIR

# Enforce cuDNN GPU acceleration
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# 1. Network Architectures
# ---------------------------------------------------------
class RoomClassifier1D(nn.Module):
    def __init__(self, in_channels: int = 256):
        super(RoomClassifier1D, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
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
        return self.classifier(feat)

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

# ---------------------------------------------------------
# 2. Episode Runner Pipeline
# ---------------------------------------------------------
class EpisodeInferencePipeline:
    def __init__(self, checkpoints_dir: Path, device: torch.device):
        self.device = device
        
        # Load Room Classifier (Task 1)
        self.room_model = RoomClassifier1D(in_channels=256).to(device)
        room_weights = checkpoints_dir / "room_classifier.pth"
        if room_weights.exists():
            self.room_model.load_state_dict(torch.load(room_weights, map_location=device))
        self.room_model.eval()

        # Load Pretrained SSL Feature Extractor (Task 2)
        self.encoder = CSIEncoderBackbone(in_channels=256, embedding_dim=128).to(device)
        ssl_weights = checkpoints_dir / "ssl_encoder.pth"
        if ssl_weights.exists():
            self.encoder.load_state_dict(torch.load(ssl_weights, map_location=device))
        self.encoder.eval()

        # Build development prototypes for shared and exclusive classes
        self.build_default_prototypes(checkpoints_dir.parent)

    def build_default_prototypes(self, workspace_dir: Path):
        manifest_path = workspace_dir / "windows/windows_manifest.csv"
        df_manifest = pd.read_csv(manifest_path)
        df_dev = df_manifest[df_manifest["split"] == "development"]

        self.prototypes = {"Room_A": {}, "Room_B": {}}

        with torch.no_grad():
            for room in ["Room_A", "Room_B"]:
                df_room = df_dev[df_dev["room"] == room]
                for act in df_room["activity"].unique():
                    files = df_room[df_room["activity"] == act]["file_path"].tolist()
                    batch_data = np.stack([np.load(f) for f in files], axis=0) # (N, 150, 256)
                    tensor_x = torch.from_numpy(batch_data.transpose(0, 2, 1)).float().to(self.device)
                    feats = self.encoder(tensor_x)
                    feats = F.normalize(feats, p=2, dim=-1)
                    proto = feats.mean(dim=0)
                    proto = F.normalize(proto, p=2, dim=0)
                    self.prototypes[room][act] = proto.cpu().numpy()

    def predict_batch(self, windows_data: np.ndarray, window_ids: list, routing_mode: str = "predicted", oracle_rooms: list = None):
        """
        Runs full pipeline over (B, 150, 256) input array.
        Output follows: window_id, room, label, confidence
        """
        B = windows_data.shape[0]
        tensor_x = torch.from_numpy(windows_data.transpose(0, 2, 1)).float().to(self.device)

        # 1. Room Routing
        predicted_rooms = []
        if routing_mode == "oracle" and oracle_rooms is not None:
            predicted_rooms = oracle_rooms
        elif routing_mode == "none":
            # Pool prototypes across both rooms without room distinction
            predicted_rooms = ["Combined"] * B
        else: # "predicted" via Task 1 classifier
            with torch.no_grad():
                room_logits = self.room_model(tensor_x)
                room_preds = torch.argmax(room_logits, dim=1).cpu().numpy()
                room_map = {0: "Room_A", 1: "Room_B"}
                predicted_rooms = [room_map[p] for p in room_preds]

        # 2. Activity Classification via Nearest Prototype
        results = []
        with torch.no_grad():
            features = self.encoder(tensor_x)
            features = F.normalize(features, p=2, dim=-1).cpu().numpy()

        for i in range(B):
            w_id = window_ids[i]
            assigned_room = predicted_rooms[i]
            feat_i = features[i]

            if assigned_room in self.prototypes:
                room_protos = self.prototypes[assigned_room]
            else:
                # Combined pool for no-routing mode
                room_protos = {**self.prototypes["Room_A"], **self.prototypes["Room_B"]}

            labels = list(room_protos.keys())
            proto_mat = np.stack([room_protos[lbl] for lbl in labels], axis=0)

            # Cosine similarities
            sims = np.dot(proto_mat, feat_i)
            # Softmax temperature-scaled confidence
            exp_sims = np.exp(sims * 10.0)
            probs = exp_sims / np.sum(exp_sims)

            best_idx = np.argmax(probs)
            best_label = labels[best_idx]
            confidence = float(probs[best_idx])

            results.append({
                "window_id": w_id,
                "room": assigned_room if assigned_room != "Combined" else "Room_A",
                "label": best_label,
                "confidence": round(confidence, 4)
            })

        return pd.DataFrame(results)

# ---------------------------------------------------------
# 3. CLI Entry Point
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-Room WiFi HAR Episode Runner")
    parser.add_argument("--input_dir", type=str, default="", help="Path to directory containing window_XXXX.npy files")
    parser.add_argument("--output_file", type=str, default="episode_predictions.csv", help="Destination CSV path")
    parser.add_argument("--routing", type=str, choices=["predicted", "oracle", "none"], default="predicted")
    args = parser.parse_args()
    checkpoints_dir = CHECKPOINTS_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Episode Inference on: {device}")

    pipeline = EpisodeInferencePipeline(checkpoints_dir, device)

    # If no custom directory passed, run against local validation set to verify
    if not args.input_dir:
        print("No input_dir provided; running reproducibility test on local validation pool...")
        manifest_path = WINDOWS_DIR / "windows_manifest.csv"
        df_val = pd.read_csv(manifest_path)
        df_val = df_val[df_val["split"] == "validation"].sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        paths = [PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p) for p in df_val["file_path"]]
        windows = np.stack([np.load(p) for p in paths], axis=0)
        window_ids = df_val["window_id"].tolist()
        oracle_rooms = df_val["room"].tolist()
    else:
        input_path = Path(args.input_dir)
        files = sorted(list(input_path.glob("*.npy")))
        if not files:
            raise FileNotFoundError(f"No .npy files found in {input_path}")
        windows = np.stack([np.load(f) for f in files], axis=0)
        window_ids = [f.stem for f in files]
        oracle_rooms = None

    preds_df = pipeline.predict_batch(
        windows_data=windows,
        window_ids=window_ids,
        routing_mode=args.routing,
        oracle_rooms=oracle_rooms
    )

    out_path = Path(args.output_file)
    preds_df.to_csv(out_path, index=False)
    print(f"\n[OK] Predictions successfully written to: {out_path.resolve()}")
    print("\nSample Output (Schema: window_id, room, label, confidence):")
    print(preds_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()