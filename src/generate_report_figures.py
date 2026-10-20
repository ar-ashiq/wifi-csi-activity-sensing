import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
checkpoints_dir = Path("D:/Ub_Hack/hack_2/checkpoints")
figures_dir = checkpoints_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Figure 1: Environment Fingerprint vs Window Length
# ---------------------------------------------------------
ablation_file = checkpoints_dir / "task1_window_ablation.csv"
if ablation_file.exists():
    df_ab = pd.read_csv(ablation_file)
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(df_ab["window_length"], df_ab["validation_accuracy"], marker='o', linewidth=2, color='#1f77b4')
    plt.title("Environment Fingerprint: Room Accuracy vs Window Length", fontsize=12, fontweight='bold')
    plt.xlabel("Observation Window Length ($T$)", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.ylim(min(df_ab["validation_accuracy"]) - 5, 102)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_environment_fingerprint.png")
    plt.close()
    print("Generated: fig1_environment_fingerprint.png")

# ---------------------------------------------------------
# Figure 2: Encoder Diagnosis t-SNE (Dual Plot)
# ---------------------------------------------------------
tsne_file = checkpoints_dir / "task2_tsne_coordinates.csv"
if tsne_file.exists():
    df_tsne = pd.read_csv(tsne_file)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    sns.scatterplot(ax=axes[0], data=df_tsne, x="tsne_dim1", y="tsne_dim2", hue="room", palette="tab10", alpha=0.8, s=35)
    axes[0].set_title("t-SNE Latent Space Colored by Room", fontweight='bold')
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")
    
    sns.scatterplot(ax=axes[1], data=df_tsne, x="tsne_dim1", y="tsne_dim2", hue="activity", palette="turbo", alpha=0.8, s=35)
    axes[1].set_title("t-SNE Latent Space Colored by Activity", fontweight='bold')
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_tsne_diagnosis.png")
    plt.close()
    print("Generated: fig2_tsne_diagnosis.png")

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