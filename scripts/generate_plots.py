"""
Generate evaluation plots and visualizations:
1. Training Loss & Accuracy Curves
2. Confusion Matrices (Model A vs Model B Baseline vs Model B Improved)
3. ROC Curves (Comparative)
Saves figures into reports/plots/
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_all_plots(
    comparison_path: str = "reports/model_comparison.json",
    history_path: str = "reports/training_history.json",
    plots_dir: str = "reports/plots"
):
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Training History Plots
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        epochs = hist.get("epochs", [])
        if epochs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Loss curve
            ax1.plot(epochs, hist.get("train_loss", []), marker="o", color="#1f77b4", label="Train Loss", linewidth=2)
            if "val_loss" in hist and hist["val_loss"]:
                ax1.plot(epochs, hist["val_loss"], marker="s", color="#ff7f0e", linestyle="--", label="Val Loss", linewidth=2)
                best_ep = hist.get("best_epoch", 5)
                ax1.axvline(best_ep, color="green", linestyle=":", label=f"Best Model (Epoch {best_ep})")
            ax1.set_title("Training & Validation Loss (BCEWithLogitsLoss)", fontsize=13, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=11)
            ax1.set_ylabel("Loss", fontsize=11)
            ax1.legend()

            # Accuracy curve
            ax2.plot(epochs, hist.get("train_accuracy", []), marker="o", color="#2ca02c", label="Train Acc (%)", linewidth=2)
            if "val_accuracy" in hist and hist["val_accuracy"]:
                ax2.plot(epochs, hist["val_accuracy"], marker="^", color="#d62728", linestyle="--", label="Val Acc (%)", linewidth=2)
            ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
            ax2.set_xlabel("Epoch", fontsize=11)
            ax2.set_ylabel("Accuracy (%)", fontsize=11)
            ax2.legend()

            plt.tight_layout()
            out_train = os.path.join(plots_dir, "training_curves.png")
            plt.savefig(out_train, dpi=300)
            plt.close()
            print(f"[Plot] Saved training curves to: {out_train}")

    # 2. Confusion Matrices & ROC Curves
    if os.path.exists(comparison_path):
        with open(comparison_path, "r", encoding="utf-8") as f:
            comp = json.load(f)

        m_a = comp["model_a_pretrained"]
        m_b_base = comp["model_b_baseline"]
        m_b_imp = comp["model_b_improved"]

        models = [
            ("Model A\n(Pretrained Baseline)", m_a),
            ("Model B\n(Colab 0.50 Baseline)", m_b_base),
            ("Model B\n(Improved Calibrated)", m_b_imp)
        ]

        # Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, (title, m_info) in zip(axes, models):
            cm_dict = m_info["confusion_matrix"]
            cm_matrix = np.array([
                [cm_dict["true_negatives_real"], cm_dict["false_positives_fake"]],
                [cm_dict["false_negatives_real"], cm_dict["true_positives_fake"]]
            ])
            sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Pred REAL", "Pred FAKE"],
                        yticklabels=["Actual REAL", "Actual FAKE"],
                        annot_kws={"size": 14, "weight": "bold"})
            acc = m_info["metrics"]["accuracy"]
            thresh = m_info["threshold_used"]
            ax.set_title(f"{title}\nAcc: {acc:.1f}% | Thresh: {thresh:.4f}", fontsize=12, fontweight="bold")

        plt.tight_layout()
        out_cm = os.path.join(plots_dir, "confusion_matrices.png")
        plt.savefig(out_cm, dpi=300)
        plt.close()
        print(f"[Plot] Saved confusion matrices to: {out_cm}")

        # ROC Curves
        plt.figure(figsize=(8, 6))
        for label, m_info, color in [
            ("Model A Baseline", m_a, "#7f7f7f"),
            ("Model B Colab Base", m_b_base, "#1f77b4"),
            ("Model B Improved", m_b_imp, "#2ca02c")
        ]:
            roc_data = m_info["roc_curve"]
            auc_val = m_info["metrics"]["roc_auc"]
            plt.plot(roc_data["fpr"], roc_data["tpr"], color=color, linewidth=2.5,
                     label=f"{label} (AUC = {auc_val:.4f})")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUC = 0.5000)")
        plt.title("Comparative ROC Curves (Locked Held-Out Test Split)", fontsize=13, fontweight="bold")
        plt.xlabel("False Positive Rate (FPR)", fontsize=11)
        plt.ylabel("True Positive Rate (TPR)", fontsize=11)
        plt.legend(loc="lower right", fontsize=11)
        plt.tight_layout()

        out_roc = os.path.join(plots_dir, "roc_curves.png")
        plt.savefig(out_roc, dpi=300)
        plt.close()
        print(f"[Plot] Saved ROC curves to: {out_roc}")


if __name__ == "__main__":
    generate_all_plots()
