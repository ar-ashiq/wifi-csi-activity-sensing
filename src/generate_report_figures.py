import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

  
from paths import CHECKPOINTS_DIR, TASK1_ABLATION_CSV


plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
checkpoints_dir = CHECKPOINTS_DIR
figures_dir = checkpoints_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Figure 3: Label-Efficiency Curve (Macro-F1 vs K)
# ---------------------------------------------------------
fewshot_file = checkpoints_dir / "task3_few_shot_results.csv"
if fewshot_file.exists():
    df_fs = pd.read_csv(fewshot_file)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)
    
    for idx, d in enumerate(["Room_A->Room_B", "Room_B->Room_A"]):
        sub = df_fs[df_fs["direction"] == d]
        axes[idx].plot(sub["K"], sub["overall_macro_f1"], marker='s', label="Overall Macro-F1", linewidth=2)
        axes[idx].plot(sub["K"], sub["exclusive_macro_f1"], marker='^', label="Target-Exclusive Activities", linewidth=2)
        axes[idx].plot(sub["K"], sub["shared_macro_f1"], marker='o', label="Transferred Shared (0-shot)", linestyle='--', linewidth=2)
        axes[idx].set_title(f"Adaptation: {d}", fontweight='bold')
        axes[idx].set_xlabel("Target Support Captures ($K$)")
        axes[idx].set_xticks([1, 5, 10, 20])
        axes[idx].set_ylabel("Macro-F1" if idx == 0 else "")
        axes[idx].grid(True, linestyle="--", alpha=0.6)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_label_efficiency_curve.png")
    plt.close()
    print("Generated: fig3_label_efficiency_curve.png")

# ---------------------------------------------------------
# Figure 4: Confusion Matrices at K=5
# ---------------------------------------------------------
cm_file = checkpoints_dir / "task3_confusion_matrices_k5.json"
if cm_file.exists():
    with open(cm_file, "r") as f:
        cm_data = json.load(f)

    for target_key in ["target_Room_B", "target_Room_A"]:
        entry = cm_data.get(target_key)
        if entry:
            labels = entry["labels"]
            matrix = np.array(entry["matrix"])
            
            plt.figure(figsize=(6.5, 5.5), dpi=300)
            sns.heatmap(matrix, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix ({target_key.replace('_', ' ')}) at K=5", fontweight='bold')
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figures_dir / f"fig4_confusion_matrix_{target_key}.png")
            plt.close()
            print(f"Generated: fig4_confusion_matrix_{target_key}.png")

print(f"\nAll report figures successfully exported to: {figures_dir.resolve()}")