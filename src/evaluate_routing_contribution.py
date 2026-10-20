import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score
import subprocess
import json

def run_routing_ablation():
    base_dir = Path("D:/Ub_Hack/hack_2")
    manifest_path = base_dir / "windows/windows_manifest.csv"
    checkpoints_dir = base_dir / "checkpoints"
    
    df_val = pd.read_csv(manifest_path)
    df_val = df_val[df_val["split"] == "validation"].copy()
    y_true = df_val["activity"].values

    routing_modes = ["predicted", "oracle", "none"]
    metrics_summary = []

    for mode in routing_modes:
        out_csv = checkpoints_dir / f"val_preds_{mode}.csv"
        cmd = [
            "python", str(base_dir / "run_episode.py"),
            "--output_file", str(out_csv),
            "--routing", mode
        ]
        subprocess.run(cmd, check=True)

        preds_df = pd.read_csv(out_csv)
        # Align with ground truth order
        merged = df_val.merge(preds_df, on="window_id")
        macro_f1 = f1_score(merged["activity"], merged["label"], average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(merged["activity"], merged["label"])

        metrics_summary.append({
            "Routing Strategy": mode.capitalize() if mode != "none" else "No Routing (Agnostic)",
            "Validation Macro-F1": round(macro_f1, 4),
            "Balanced Accuracy (%)": round(bal_acc * 100, 2)
        })

    summary_df = pd.DataFrame(metrics_summary)
    summary_path = checkpoints_dir / "routing_contribution_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("Routing Stage Contribution Benchmark:")
    print(summary_df.to_string(index=False))
    print("=" * 65)
    print(f"Saved routing ablation data to: {summary_path.resolve()}")

if __name__ == "__main__":
    run_routing_ablation()