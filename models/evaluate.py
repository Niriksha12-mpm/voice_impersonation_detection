"""
Rigorous Evaluation and Model Comparison Engine.
Evaluates Model A (Pretrained Model) and Model B (Fine-Tuned Model)
on the exact same held-out test split (and cross-dataset evaluation set).

Calculates genuine, experimental metrics without fabrication:
- Accuracy (%)
- Precision (%)
- Recall (%)
- F1-score (%)
- Confusion Matrix: True Negatives, False Positives, False Negatives, True Positives
- ROC-AUC Score
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Inference Latency (mean ms, std ms) and throughput (samples/sec)
- Class-wise and Dataset-wise breakdowns
- Cross-dataset out-of-distribution evaluation
Saves full report to reports/evaluation_metrics.json and reports/model_comparison.json.
"""

import os
import json
import time
import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from models.architecture import DeepfakeAudioClassifier, load_classifier
from models.feature_extractor import MelSpectrogramExtractor


def evaluate_single_model(
    model: DeepfakeAudioClassifier,
    manifest_records: list,
    model_name: str,
    extractor: MelSpectrogramExtractor,
    device: torch.device
) -> dict:
    """Evaluate a model on a set of manifest records and compute full metrics."""
    y_true = []
    y_pred = []
    y_probs = []
    latencies = []

    per_sample_results = []
    dataset_breakdown: dict = {}

    model.eval()

    with torch.no_grad():
        for rec in manifest_records:
            file_path = rec["segment_path"]
            ground_truth = rec["label"]
            y_true_binary = 1.0 if ground_truth == "FAKE" else 0.0

            # Measure inference latency
            t0 = time.perf_counter()
            y, sr = sf.read(file_path, dtype="float32")
            tensor = extractor.extract_tensor(y).to(device)
            logit = model(tensor).item()
            prob_fake = 1.0 / (1.0 + np.exp(-logit))
            latency_ms = (time.perf_counter() - t0) * 1000.0

            predicted_label = "FAKE" if prob_fake >= 0.5 else "REAL"
            y_pred_binary = 1.0 if predicted_label == "FAKE" else 0.0

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
                "filename": rec["segment_filename"],
                "parent_file": rec["parent_file"],
                "dataset_source": ds_src,
                "ground_truth": ground_truth,
                "predicted_label": predicted_label,
                "probability_fake": round(float(prob_fake), 4),
                "is_correct": is_correct,
                "latency_ms": round(float(latency_ms), 2)
            })

    # Metric computations
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0

    # ROC AUC
    try:
        if len(set(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_probs))
            fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_probs)
            # scikit-learn sets thresholds[0] to inf; replace with 1.0 for JSON compliance
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
    cm = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # FPR and FNR
    fpr = float(fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) * 100.0 if (fn + tp) > 0 else 0.0

    # Class-wise metrics
    real_precision = float(tn / (tn + fn)) * 100.0 if (tn + fn) > 0 else 0.0
    real_recall = float(tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    real_f1 = (2 * real_precision * real_recall) / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0.0

    # Dataset accuracy breakdown
    ds_summary = {}
    for ds_name, stats in dataset_breakdown.items():
        ds_summary[ds_name] = {
            "total_samples": stats["total"],
            "correct_samples": stats["correct"],
            "accuracy_percent": round(stats["correct"] / stats["total"] * 100.0, 2),
            "real_count": stats["real"],
            "fake_count": stats["fake"]
        }

    # Efficiency
    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    throughput = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

    results = {
        "model_name": model_name,
        "sample_count": len(manifest_records),
        "metrics": {
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 2),
            "false_negative_rate": round(fnr, 2)
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
    return results


def run_full_evaluation(
    test_manifest_path: str = "data/splits/test_manifest.json",
    cross_manifest_path: str = "data/splits/cross_dataset_manifest.json",
    pretrained_weights: str = "models/saved/pretrained_model_a.pt",
    fine_tuned_weights: str = "models/saved/fine_tuned_model_b.pt",
    output_comparison: str = "reports/model_comparison.json",
    output_metrics: str = "reports/evaluation_metrics.json"
) -> dict:
    """Compare Model A vs Model B on held-out test split and cross-dataset split."""
    device = torch.device("cpu")
    extractor = MelSpectrogramExtractor()

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    cross_records = []
    if os.path.exists(cross_manifest_path):
        with open(cross_manifest_path, "r", encoding="utf-8") as f:
            cross_records = json.load(f)

    print(f"[Evaluate] Running honest evaluation on {len(test_records)} held-out test samples...")

    # Load Model A
    model_a = load_classifier(pretrained_weights, map_location="cpu")
    model_a.to(device)

    # Load Model B
    model_b = load_classifier(fine_tuned_weights, map_location="cpu")
    model_b.to(device)

    # Evaluate Model A on Test Split
    eval_a = evaluate_single_model(model_a, test_records, "Model A (Pretrained Model)", extractor, device)

    # Evaluate Model B on Test Split
    eval_b = evaluate_single_model(model_b, test_records, "Model B (Fine-Tuned Model)", extractor, device)

    # Cross-dataset evaluation if samples available
    cross_eval_a = None
    cross_eval_b = None
    if cross_records:
        print(f"[Evaluate] Evaluating on {len(cross_records)} cross-dataset out-of-distribution samples...")
        cross_eval_a = evaluate_single_model(model_a, cross_records, "Model A (Cross-Dataset)", extractor, device)
        cross_eval_b = evaluate_single_model(model_b, cross_records, "Model B (Cross-Dataset)", extractor, device)

    # Compute deltas
    delta = {
        "accuracy_delta": round(eval_b["metrics"]["accuracy"] - eval_a["metrics"]["accuracy"], 2),
        "precision_delta": round(eval_b["metrics"]["precision"] - eval_a["metrics"]["precision"], 2),
        "recall_delta": round(eval_b["metrics"]["recall"] - eval_a["metrics"]["recall"], 2),
        "f1_score_delta": round(eval_b["metrics"]["f1_score"] - eval_a["metrics"]["f1_score"], 2),
        "roc_auc_delta": round(eval_b["metrics"]["roc_auc"] - eval_a["metrics"]["roc_auc"], 4),
        "fpr_reduction": round(eval_a["metrics"]["false_positive_rate"] - eval_b["metrics"]["false_positive_rate"], 2),
        "fnr_reduction": round(eval_a["metrics"]["false_negative_rate"] - eval_b["metrics"]["false_negative_rate"], 2),
        "latency_overhead_ms": round(eval_b["efficiency"]["avg_latency_ms"] - eval_a["efficiency"]["avg_latency_ms"], 2)
    }

    comparison_report = {
        "timestamp": "2026-10-14",
        "held_out_test_samples": len(test_records),
        "held_out_distribution": {
            "real": sum(1 for x in test_records if x["label"] == "REAL"),
            "fake": sum(1 for x in test_records if x["label"] == "FAKE")
        },
        "model_a_pretrained": eval_a,
        "model_b_fine_tuned": eval_b,
        "improvement_delta": delta,
        "cross_dataset_evaluation": {
            "available": len(cross_records) > 0,
            "sample_count": len(cross_records),
            "model_a_cross_metrics": cross_eval_a["metrics"] if cross_eval_a else None,
            "model_b_cross_metrics": cross_eval_b["metrics"] if cross_eval_b else None,
            "analysis": "Evaluates domain shift generalization on speech generated from engines not present during training."
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_comparison)), exist_ok=True)
    with open(output_comparison, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump({
            "test_evaluation": {
                "model_a": eval_a["metrics"],
                "model_b": eval_b["metrics"],
                "delta": delta
            },
            "confusion_matrices": {
                "model_a": eval_a["confusion_matrix"],
                "model_b": eval_b["confusion_matrix"]
            },
            "efficiency": {
                "model_a": eval_a["efficiency"],
                "model_b": eval_b["efficiency"]
            }
        }, f, indent=2)

    print(f"\n=======================================================")
    print(f"               EXPERIMENTAL EVALUATION RESULTS          ")
    print(f"=======================================================")
    print(f"Metric                 Model A (Pretrained)   Model B (Fine-Tuned)   Delta")
    print(f"-------------------------------------------------------")
    print(f"Accuracy:              {eval_a['metrics']['accuracy']:>6.2f}%               {eval_b['metrics']['accuracy']:>6.2f}%             {delta['accuracy_delta']:>+6.2f}%")
    print(f"Precision:             {eval_a['metrics']['precision']:>6.2f}%               {eval_b['metrics']['precision']:>6.2f}%             {delta['precision_delta']:>+6.2f}%")
    print(f"Recall:                {eval_a['metrics']['recall']:>6.2f}%               {eval_b['metrics']['recall']:>6.2f}%             {delta['recall_delta']:>+6.2f}%")
    print(f"F1-Score:              {eval_a['metrics']['f1_score']:>6.2f}%               {eval_b['metrics']['f1_score']:>6.2f}%             {delta['f1_score_delta']:>+6.2f}%")
    print(f"ROC-AUC:               {eval_a['metrics']['roc_auc']:>6.4f}                {eval_b['metrics']['roc_auc']:>6.4f}              {delta['roc_auc_delta']:>+6.4f}")
    print(f"False Positive Rate:   {eval_a['metrics']['false_positive_rate']:>6.2f}%               {eval_b['metrics']['false_positive_rate']:>6.2f}%             {-delta['fpr_reduction']:>+6.2f}%")
    print(f"False Negative Rate:   {eval_a['metrics']['false_negative_rate']:>6.2f}%               {eval_b['metrics']['false_negative_rate']:>6.2f}%             {-delta['fnr_reduction']:>+6.2f}%")
    print(f"Avg Latency:           {eval_a['efficiency']['avg_latency_ms']:>6.2f} ms             {eval_b['efficiency']['avg_latency_ms']:>6.2f} ms           {delta['latency_overhead_ms']:>+6.2f} ms")
    print(f"=======================================================")
    print(f"[Evaluate] Comparison saved to: {output_comparison}")
    print(f"[Evaluate] Metrics saved to: {output_metrics}")
    return comparison_report


if __name__ == "__main__":
    run_full_evaluation()
