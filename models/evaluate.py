"""
Rigorous Evaluation and Model Comparison Engine.
Evaluates:
1. Model A (Pretrained CNN Baseline)
2. Model B Baseline (Colab Lakhs weights with default 0.5 threshold)
3. Model B Improved (Fine-Tuned Four-Block CNN-BiLSTM with calibrated validation threshold)

Evaluated on the exact same held-out test split (and cross-dataset evaluation set).
Computes genuine, non-fabricated metrics:
- Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR), False Negative Rate (FNR), Equal Error Rate (EER)
- Inference Latency and Throughput
- 95% Bootstrap Confidence Intervals for Accuracy
- Outputs JSON and CSV reports into reports/
"""

import os
import json
import csv
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from models.architecture import DeepfakeCNNBiLSTMClassifier, load_classifier
from models.feature_extractor import AudioFeatureExtractor


def compute_bootstrap_ci(y_true: List[int], y_pred: List[int], n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for test accuracy."""
    np.random.seed(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    scores = []
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, size=n)
        if len(set(y_t[indices])) < 2:
            scores.append(float(np.mean(y_t[indices] == y_p[indices])))
        else:
            scores.append(accuracy_score(y_t[indices], y_p[indices]))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100)) * 100.0
    upper = float(np.percentile(scores, (1.0 - alpha) * 100)) * 100.0
    return round(lower, 2), round(upper, 2)


def compute_eer(y_true: List[int], y_probs: List[float]) -> float:
    """Compute Equal Error Rate (point where FPR == FNR)."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        return round(float((fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100.0), 2)
    except Exception:
        return 50.0


def evaluate_single_model(
    model: torch.nn.Module,
    manifest_records: list,
    model_name: str,
    extractor: AudioFeatureExtractor,
    device: torch.device,
    custom_threshold: Optional[float] = None
) -> dict:
    """Evaluate a model on a set of manifest records and compute full metrics."""
    y_true = []
    y_pred = []
    y_probs = []
    latencies = []

    per_sample_results = []
    dataset_breakdown: dict = {}

    threshold = custom_threshold if custom_threshold is not None else getattr(model, "decision_threshold", 0.5)
    model.eval()

    with torch.no_grad():
        for rec in manifest_records:
            file_path = rec["segment_path"]
            ground_truth = rec["label"].strip().upper()
            y_true_binary = 1 if ground_truth == "FAKE" else 0

            # Measure latency
            t0 = time.perf_counter()
            y, sr = sf.read(file_path, dtype="float32")
            tensor = extractor.extract_tensor(y).to(device)
            logit = model(tensor).item()
            prob_fake = 1.0 / (1.0 + np.exp(-logit))
            latency_ms = (time.perf_counter() - t0) * 1000.0

            predicted_label = "FAKE" if prob_fake >= threshold else "REAL"
            y_pred_binary = 1 if predicted_label == "FAKE" else 0

            y_true.append(y_true_binary)
            y_pred.append(y_pred_binary)
            y_probs.append(prob_fake)
            latencies.append(latency_ms)

            is_correct = (predicted_label == ground_truth)
            ds_src = rec.get("dataset_source", "Unknown")

            dataset_breakdown.setdefault(ds_src, {"total": 0, "correct": 0, "real": 0, "fake": 0})
            dataset_breakdown[ds_src]["total"] += 1
            if is_correct:
                dataset_breakdown[ds_src]["correct"] += 1
            if ground_truth == "REAL":
                dataset_breakdown[ds_src]["real"] += 1
            else:
                dataset_breakdown[ds_src]["fake"] += 1

            per_sample_results.append({
                "filename": rec.get("segment_filename", os.path.basename(file_path)),
                "parent_file": rec.get("parent_file", ""),
                "dataset_source": ds_src,
                "ground_truth": ground_truth,
                "predicted_label": predicted_label,
                "probability_fake": round(float(prob_fake), 4),
                "is_correct": is_correct,
                "latency_ms": round(float(latency_ms), 2)
            })

    # Metric computations
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    ci_lower, ci_upper = compute_bootstrap_ci(y_true, y_pred)
    eer = compute_eer(y_true, y_probs)

    # ROC AUC
    try:
        if len(set(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_probs))
            fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_probs)
            safe_thresh = [1.0 if np.isinf(x) else round(float(x), 4) for x in thresholds]
            roc_curve_data = {
                "fpr": [round(float(x), 4) for x in fpr_vals],
                "tpr": [round(float(x), 4) for x in tpr_vals],
                "thresholds": safe_thresh
            }
        else:
            roc_auc = 0.5
            roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}
    except Exception:
        roc_auc = 0.5
        roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}

    # Confusion matrix: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = float(fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) * 100.0 if (fn + tp) > 0 else 0.0

    real_precision = float(tn / (tn + fn)) * 100.0 if (tn + fn) > 0 else 0.0
    real_recall = float(tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    real_f1 = (2 * real_precision * real_recall) / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0.0

    ds_summary = {}
    for ds_name, stats in dataset_breakdown.items():
        ds_summary[ds_name] = {
            "total_samples": stats["total"],
            "correct_samples": stats["correct"],
            "accuracy_percent": round(stats["correct"] / stats["total"] * 100.0, 2),
            "real_count": stats["real"],
            "fake_count": stats["fake"]
        }

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    throughput = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

    return {
        "model_name": model_name,
        "sample_count": len(manifest_records),
        "threshold_used": round(threshold, 4),
        "metrics": {
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bal_acc, 2),
            "accuracy_95ci": [ci_lower, ci_upper],
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 2),
            "false_negative_rate": round(fnr, 2),
            "equal_error_rate": round(eer, 2)
        },
        "confusion_matrix": {
            "true_negatives_real": tn,
            "false_positives_fake": fp,
            "false_negatives_real": fn,
            "true_positives_fake": tp
        },
        "class_wise_metrics": {
            "real": {
                "precision": round(real_precision, 2),
                "recall": round(real_recall, 2),
                "f1_score": round(real_f1, 2),
                "support": tn + fp
            },
            "fake": {
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1_score": round(f1, 2),
                "support": tp + fn
            }
        },
        "efficiency": {
            "avg_latency_ms": round(avg_latency, 2),
            "std_latency_ms": round(std_latency, 2),
            "throughput_samples_per_sec": throughput,
            "parameter_count": model.count_parameters()
        },
        "dataset_accuracy_breakdown": ds_summary,
        "roc_curve": roc_curve_data,
        "per_sample_details": per_sample_results
    }


def export_reports_to_csv(comparison_report: dict, base_dir: str = "reports"):
    """Export evaluation metrics to structured CSV files."""
    csv_path = os.path.join(base_dir, "model_comparison.csv")
    metrics_csv_path = os.path.join(base_dir, "evaluation_metrics.csv")

    rows = []
    models_to_export = [
        ("Model A (Pretrained Baseline)", comparison_report["model_a_pretrained"]),
        ("Model B (Colab Baseline)", comparison_report["model_b_baseline"]),
        ("Model B (Improved Fine-Tuned)", comparison_report["model_b_improved"])
    ]

    for model_label, m_data in models_to_export:
        met = m_data["metrics"]
        cm = m_data["confusion_matrix"]
        eff = m_data["efficiency"]
        rows.append({
            "Model": model_label,
            "Threshold": m_data["threshold_used"],
            "Accuracy (%)": met["accuracy"],
            "Balanced Accuracy (%)": met["balanced_accuracy"],
            "95% CI Lower": met["accuracy_95ci"][0],
            "95% CI Upper": met["accuracy_95ci"][1],
            "Precision (%)": met["precision"],
            "Recall (%)": met["recall"],
            "F1-Score (%)": met["f1_score"],
            "ROC-AUC": met["roc_auc"],
            "FPR (%)": met["false_positive_rate"],
            "FNR (%)": met["false_negative_rate"],
            "EER (%)": met["equal_error_rate"],
            "TN": cm["true_negatives_real"],
            "FP": cm["false_positives_fake"],
            "FN": cm["false_negatives_real"],
            "TP": cm["true_positives_fake"],
            "Avg Latency (ms)": eff["avg_latency_ms"],
            "Throughput (samples/s)": eff["throughput_samples_per_sec"],
            "Parameters": eff["parameter_count"]
        })

    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Export] Saved CSV comparison to: {csv_path}")
    print(f"[Export] Saved CSV metrics to:    {metrics_csv_path}")


def run_full_evaluation(
    test_manifest_path: str = "data/splits/test_manifest.json",
    cross_manifest_path: str = "data/splits/cross_dataset_manifest.json",
    pretrained_weights: str = "models/saved/pretrained_model_a.pt",
    colab_weights: str = "models/saved/fine_tuned_model_b_lakhs.pt",
    improved_weights: str = "models/saved/fine_tuned_model_b.pt",
    output_comparison: str = "reports/model_comparison.json",
    output_metrics: str = "reports/evaluation_metrics.json"
) -> dict:
    """Compare Model A vs Model B Baseline vs Model B Improved on held-out test split."""
    device = torch.device("cpu")
    extractor = AudioFeatureExtractor()

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    cross_records = []
    if os.path.exists(cross_manifest_path):
        with open(cross_manifest_path, "r", encoding="utf-8") as f:
            cross_records = json.load(f)

    print(f"[Evaluate] Running rigorous evaluation on {len(test_records)} held-out test samples...")

    # Load Model A (Pretrained Pure CNN)
    model_a = load_classifier(pretrained_weights, map_location="cpu")
    model_a.to(device)

    # Load Model B Baseline (Colab Lakhs)
    model_b_baseline = load_classifier(colab_weights, map_location="cpu")
    model_b_baseline.to(device)

    # Load Model B Improved (Fine-Tuned)
    model_b_improved = load_classifier(improved_weights, map_location="cpu")
    model_b_improved.to(device)

    # 1. Evaluate Model A on Test Split (Threshold = 0.5)
    eval_a = evaluate_single_model(model_a, test_records, "Model A (Pretrained Baseline)", extractor, device, custom_threshold=0.5)

    # 2. Evaluate Model B Baseline on Test Split (Threshold = 0.5)
    eval_b_base = evaluate_single_model(model_b_baseline, test_records, "Model B (Colab Baseline)", extractor, device, custom_threshold=0.5)

    # 3. Evaluate Model B Improved on Test Split (Calibrated Validation Threshold)
    eval_b_imp = evaluate_single_model(model_b_improved, test_records, "Model B (Improved Fine-Tuned)", extractor, device)

    # Cross-dataset evaluation
    cross_eval_a = None
    cross_eval_b = None
    if cross_records:
        cross_eval_a = evaluate_single_model(model_a, cross_records, "Model A (Cross-Dataset)", extractor, device, custom_threshold=0.5)
        cross_eval_b = evaluate_single_model(model_b_improved, cross_records, "Model B (Cross-Dataset)", extractor, device)

    # Calculate Deltas over Colab Baseline
    acc_base = eval_b_base["metrics"]["accuracy"]
    acc_imp = eval_b_imp["metrics"]["accuracy"]
    acc_delta_pp = round(acc_imp - acc_base, 2)
    acc_rel_change = round((acc_imp - acc_base) / max(acc_base, 1e-5) * 100.0, 2)

    # Count changed predictions between baseline and improved
    changed_preds = sum(
        1 for sa, sb in zip(eval_b_base["per_sample_details"], eval_b_imp["per_sample_details"])
        if sa["predicted_label"] != sb["predicted_label"]
    )

    delta = {
        "baseline_model": "Model B (Colab Baseline)",
        "improved_model": "Model B (Improved Fine-Tuned)",
        "accuracy_baseline": acc_base,
        "accuracy_improved": acc_imp,
        "accuracy_delta_percentage_points": acc_delta_pp,
        "accuracy_relative_change_percent": acc_rel_change,
        "changed_predictions_count": changed_preds,
        "precision_delta": round(eval_b_imp["metrics"]["precision"] - eval_b_base["metrics"]["precision"], 2),
        "recall_delta": round(eval_b_imp["metrics"]["recall"] - eval_b_base["metrics"]["recall"], 2),
        "f1_score_delta": round(eval_b_imp["metrics"]["f1_score"] - eval_b_base["metrics"]["f1_score"], 2),
        "fpr_reduction": round(eval_b_base["metrics"]["false_positive_rate"] - eval_b_imp["metrics"]["false_positive_rate"], 2),
        "fnr_reduction": round(eval_b_base["metrics"]["false_negative_rate"] - eval_b_imp["metrics"]["false_negative_rate"], 2),
        "meets_target_improvement": bool(acc_delta_pp >= 0.1)
    }

    comparison_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "held_out_test_samples": len(test_records),
        "held_out_distribution": {
            "real": sum(1 for x in test_records if x["label"] == "REAL"),
            "fake": sum(1 for x in test_records if x["label"] == "FAKE")
        },
        "model_a_pretrained": eval_a,
        "model_b_baseline": eval_b_base,
        "model_b_improved": eval_b_imp,
        "improvement_delta": delta,
        "cross_dataset_evaluation": {
            "available": len(cross_records) > 0,
            "sample_count": len(cross_records),
            "model_a_cross_metrics": cross_eval_a["metrics"] if cross_eval_a else None,
            "model_b_cross_metrics": cross_eval_b["metrics"] if cross_eval_b else None,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_comparison)), exist_ok=True)
    with open(output_comparison, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump({
            "test_evaluation": {
                "model_a_pretrained": eval_a["metrics"],
                "model_b_baseline": eval_b_base["metrics"],
                "model_b_improved": eval_b_imp["metrics"],
                "delta": delta
            },
            "confusion_matrices": {
                "model_a_pretrained": eval_a["confusion_matrix"],
                "model_b_baseline": eval_b_base["confusion_matrix"],
                "model_b_improved": eval_b_imp["confusion_matrix"]
            },
            "efficiency": {
                "model_a_pretrained": eval_a["efficiency"],
                "model_b_baseline": eval_b_base["efficiency"],
                "model_b_improved": eval_b_imp["efficiency"]
            }
        }, f, indent=2)

    export_reports_to_csv(comparison_report)

    print(f"\n" + "=" * 70)
    print(f"             VOICEGUARD AI: MODEL EVALUATION BENCHMARK")
    print(f"=" * 70)
    print(f"{'Metric':<24} {'Model A (Pretrained)':<20} {'Model B (Colab Base)':<20} {'Model B (Improved)':<20}")
    print(f"-" * 70)
    print(f"{'Accuracy':<24} {eval_a['metrics']['accuracy']:>5.2f}%              {eval_b_base['metrics']['accuracy']:>5.2f}%              {eval_b_imp['metrics']['accuracy']:>5.2f}%")
    print(f"{'Balanced Accuracy':<24} {eval_a['metrics']['balanced_accuracy']:>5.2f}%              {eval_b_base['metrics']['balanced_accuracy']:>5.2f}%              {eval_b_imp['metrics']['balanced_accuracy']:>5.2f}%")
    print(f"{'Precision':<24} {eval_a['metrics']['precision']:>5.2f}%              {eval_b_base['metrics']['precision']:>5.2f}%              {eval_b_imp['metrics']['precision']:>5.2f}%")
    print(f"{'Recall':<24} {eval_a['metrics']['recall']:>5.2f}%              {eval_b_base['metrics']['recall']:>5.2f}%              {eval_b_imp['metrics']['recall']:>5.2f}%")
    print(f"{'F1-Score':<24} {eval_a['metrics']['f1_score']:>5.2f}%              {eval_b_base['metrics']['f1_score']:>5.2f}%              {eval_b_imp['metrics']['f1_score']:>5.2f}%")
    print(f"{'ROC-AUC':<24} {eval_a['metrics']['roc_auc']:>6.4f}               {eval_b_base['metrics']['roc_auc']:>6.4f}               {eval_b_imp['metrics']['roc_auc']:>6.4f}")
    print(f"{'False Positive Rate':<24} {eval_a['metrics']['false_positive_rate']:>5.2f}%              {eval_b_base['metrics']['false_positive_rate']:>5.2f}%              {eval_b_imp['metrics']['false_positive_rate']:>5.2f}%")
    print(f"{'False Negative Rate':<24} {eval_a['metrics']['false_negative_rate']:>5.2f}%              {eval_b_base['metrics']['false_negative_rate']:>5.2f}%              {eval_b_imp['metrics']['false_negative_rate']:>5.2f}%")
    print(f"{'Decision Threshold':<24} {eval_a['threshold_used']:>6.4f}               {eval_b_base['threshold_used']:>6.4f}               {eval_b_imp['threshold_used']:>6.4f}")
    print(f"=" * 70)
    print(f"Target Improvement: +0.10 percentage points")
    print(f"Achieved Improvement: {delta['accuracy_delta_percentage_points']:+6.2f} percentage points ({delta['accuracy_relative_change_percent']:+6.2f}% relative)")
    print(f"Changed Predictions: {delta['changed_predictions_count']} out of {len(test_records)} test samples")
    print(f"Target Met: {'[PASS] Target Exceeded' if delta['meets_target_improvement'] else '[FAIL]'}")
    print(f"=" * 70)

    return comparison_report


if __name__ == "__main__":
    run_full_evaluation()
