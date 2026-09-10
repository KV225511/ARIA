import json
import os
import logging
from pathlib import Path
import numpy as np
from collections import Counter
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

logger = logging.getLogger(__name__)

METRICS_SCHEMA_VERSION = "aria-classification-metrics-v2"
CLASS_LABELS = (0, 1, 2)


def compute_classification_metrics(true_labels, decision_predictions, class_probabilities=None, bins=10):
    """Canonical abstention-aware classification metrics for ARIA."""
    if len(true_labels) != len(decision_predictions):
        raise ValueError("true_labels and decision_predictions must have equal length")
    if not true_labels:
        return {}
    if any(label not in CLASS_LABELS for label in true_labels):
        raise ValueError("true_labels must contain only 0, 1, or 2")
    predictions = []
    for prediction in decision_predictions:
        if prediction in CLASS_LABELS:
            predictions.append(int(prediction))
        elif prediction is None or prediction == -1:
            predictions.append(-1)
        else:
            raise ValueError("prediction must be 0, 1, 2, None, or -1")
    total = len(true_labels)
    classified = [(truth, prediction) for truth, prediction in zip(true_labels, predictions) if prediction != -1]
    counts = Counter(prediction for _, prediction in classified)
    confusion = [[0, 0, 0, 0] for _ in CLASS_LABELS]
    for truth, prediction in zip(true_labels, predictions):
        confusion[int(truth)][3 if prediction == -1 else prediction] += 1
    per_class, f1s, recalls = {}, [], []
    for label in CLASS_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[truth][label] for truth in CLASS_LABELS if truth != label)
        fn = sum(confusion[label][column] for column in range(4) if column != label)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
        recalls.append(recall)
    num_classified = len(classified)
    correct = sum(truth == prediction for truth, prediction in classified)
    result = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "num_examples": total,
        "num_classified": num_classified,
        "num_abstained": total - num_classified,
        "coverage": num_classified / total,
        "abstention_rate": (total - num_classified) / total,
        "overall_accuracy": correct / total,
        "selective_accuracy": correct / num_classified if num_classified else 0.0,
        "selective_risk": 1.0 - (correct / num_classified if num_classified else 0.0),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "minimum_class_recall": float(min(recalls)),
        "maximum_classified_prediction_share": max(counts.values()) / num_classified if num_classified else 1.0,
        "ordinal_mae": float(np.mean([2.0 if p == -1 else abs(int(t) - p) for t, p in zip(true_labels, predictions)])),
        "true_label_counts": dict(Counter(true_labels)),
        "decision_prediction_counts": dict(counts),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "accuracy": correct / total,
        "micro_f1": correct / total,
        "terminal_micro_f1": correct / total,
    }
    if class_probabilities is None:
        return {**result, "brier_score": None, "expected_calibration_error": None, "argmax_prediction_counts": {}}
    probabilities = np.asarray(class_probabilities, dtype=float)
    if probabilities.shape != (total, 3) or not np.all(np.isfinite(probabilities)):
        raise ValueError("class_probabilities must be finite with shape (N, 3)")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("class_probabilities must be non-negative and sum to one")
    argmax = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    argmax_correct = (argmax == np.asarray(true_labels, dtype=int)).astype(float)
    ece, edges = 0.0, np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(confidence[mask].mean()) - float(argmax_correct[mask].mean()))
    one_hot = np.eye(3)[np.asarray(true_labels, dtype=int)]
    return {**result,
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(ece),
        "argmax_prediction_counts": dict(Counter(int(value) for value in argmax)),
    }

def compute_rl_metrics(episodes_data):
    """Describe fixed logged trajectories; do not imply policy evaluation."""
    if not episodes_data:
        return {}
        
    lengths = [len(ep) for ep in episodes_data]
    avg_length = np.mean(lengths)
    
    entropies = []
    rewards = []
    
    for ep in episodes_data:
        rewards.append(sum(t.get('reward', 0) for t in ep))
        
        # Policy entropy: calculate from action distribution
        actions = [t['action_idx'] for t in ep if 'action_idx' in t]
        if actions:
            _, counts = np.unique(actions, return_counts=True)
            probs = counts / len(actions)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            entropies.append(entropy)
        else:
            entropies.append(0.0)
            
    return {
        "evaluation_type": "logged_behavior_trajectories",
        "evaluates_learned_policy": False,
        "logged_avg_episode_length": float(avg_length),
        "logged_avg_cumulative_reward": float(np.mean(rewards)),
        "logged_action_entropy": float(np.mean(entropies)),
    }

def compute_response_metrics(
    aria_labels,
    true_labels,
    questions=None,
    jd_text="",
    beliefs=None,
):
    """Compatibility wrapper around the canonical classification contract."""
    if not aria_labels or not true_labels:
        return {}
    probabilities = beliefs
    if probabilities is not None:
        try:
            array = np.asarray(probabilities, dtype=float)
            if array.shape != (len(true_labels), 3) or not np.all(np.isfinite(array)):
                probabilities = None
        except (TypeError, ValueError):
            probabilities = None
    result = compute_classification_metrics(true_labels, aria_labels, probabilities)
    # Backward-compatible name used by existing reports and callers.
    result["abstention_count"] = result["num_abstained"]
    result["aria_label_counts"] = dict(Counter(aria_labels))
    result["rouge_L"] = 0.0
    return result
        
    # Assuming labels are categorical (0: beginner, 1: mid, 2: expert)
    normalized_predictions = [label if label in (0, 1, 2) else -1 for label in aria_labels]
    labels = [0, 1, 2]
    acc = accuracy_score(true_labels, normalized_predictions)
    macro_f1 = f1_score(true_labels, normalized_predictions, labels=labels, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, normalized_predictions, labels=labels, average='micro', zero_division=0)
    precision = precision_score(true_labels, normalized_predictions, labels=labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, normalized_predictions, labels=labels, average='macro', zero_division=0)
    kappa = (
        0.0
        if len(set(true_labels) | set(normalized_predictions)) < 2
        else cohen_kappa_score(true_labels, normalized_predictions)
    )
    conf_mat = confusion_matrix(true_labels, normalized_predictions, labels=labels).tolist()
    ordinal_errors = [
        2 if predicted == -1 else abs(int(truth) - int(predicted))
        for truth, predicted in zip(true_labels, normalized_predictions)
    ]

    brier_score = None
    expected_calibration_error = None
    if beliefs and len(beliefs) == len(true_labels):
        belief_array = np.asarray(beliefs, dtype=float)
        if belief_array.shape == (len(true_labels), 3) and np.all(np.isfinite(belief_array)):
            one_hot = np.eye(3)[np.asarray(true_labels, dtype=int)]
            brier_score = float(np.mean(np.sum((belief_array - one_hot) ** 2, axis=1)))
            confidences = np.max(belief_array, axis=1)
            correct = np.asarray([
                truth == prediction
                for truth, prediction in zip(true_labels, normalized_predictions)
            ], dtype=float)
            expected_calibration_error = 0.0
            edges = np.linspace(0.0, 1.0, 11)
            for index in range(10):
                mask = (confidences >= edges[index]) & (
                    confidences <= edges[index + 1]
                    if index == 9 else confidences < edges[index + 1]
                )
                if np.any(mask):
                    expected_calibration_error += float(np.mean(mask)) * abs(
                        float(np.mean(confidences[mask])) - float(np.mean(correct[mask]))
                    )
    
    avg_rougeL = 0.0
    if questions and jd_text and rouge_scorer is not None:
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = []
            for q in questions:
                score = scorer.score(jd_text, q)
                rouge_scores.append(score['rougeL'].fmeasure)
            avg_rougeL = float(np.mean(rouge_scores)) if rouge_scores else 0.0
        except Exception:
            avg_rougeL = 0.0
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "balanced_accuracy": float(recall),
        "cohens_kappa": float(kappa),
        "confusion_matrix": conf_mat,
        "ordinal_mae": float(np.mean(ordinal_errors)),
        "abstention_count": int(sum(label == -1 for label in normalized_predictions)),
        "brier_score": brier_score,
        "expected_calibration_error": expected_calibration_error,
        "true_label_counts": dict(Counter(true_labels)),
        "aria_label_counts": dict(Counter(aria_labels)),
        "rouge_L": avg_rougeL
    }

def debug_eval(name: str, y_true, y_pred, labels=None):
    """Prints a clean diagnostic breakdown for benchmark evaluations."""
    print(f"\n=== {name} DIAGNOSTIC ===")
    print("num_samples:", len(y_true))
    print("y_true counts:", dict(Counter(y_true)))
    print("y_pred counts:", dict(Counter(y_pred)))
    if len(y_true) > 0 and len(y_pred) > 0:
        print("confusion_matrix (labels=[0, 1, 2]):")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
        print("classification_report:")
        print(classification_report(y_true, y_pred, labels=[0, 1, 2], zero_division=0, target_names=["Beginner", "Mid", "Expert"]))

def build_belief_report(dataset: list[dict], emit_debug: bool = False):
    """Build metrics for stored Bayesian verdicts, never a learned policy."""
    if not dataset:
        return {}

    episodes = group_transitions_into_episodes(dataset)
    
    # Terminal (final conclusion) labels
    terminal_aria_labels = []
    terminal_true_labels = []
    terminal_beliefs = []
    
    # All transitions labels
    all_aria_labels = []
    all_true_labels = []
    
    questions = []
    jd_texts = []
    
    for t in dataset:
        # Collect overall labels
        if "true_label" in t and "aria_label" in t:
            all_true_labels.append(t["true_label"])
            all_aria_labels.append(t["aria_label"])
            
        if "question" in t and "jd_text" in t:
            questions.append(t["question"])
            if t["jd_text"] not in jd_texts:
                jd_texts.append(t["jd_text"])
                
    for episode in episodes:
        if not episode:
            continue
        terminal = episode[-1]
        if "true_label" in terminal:
            terminal_true_labels.append(terminal["true_label"])
            terminal_aria_labels.append(terminal.get("aria_label"))
            terminal_beliefs.append(terminal.get("aggregate_belief"))
        
    rl_metrics = compute_rl_metrics(episodes)
    
    jd_text_combined = " ".join(jd_texts)
    
    # Evaluate terminal decisions (honest assessment of interview outcome)
    terminal_response_metrics = compute_response_metrics(
        terminal_aria_labels,
        terminal_true_labels,
        questions,
        jd_text_combined,
        beliefs=terminal_beliefs,
    )
    
    # Also record overall transitions response metrics
    overall_response_metrics = compute_response_metrics(
        all_aria_labels, all_true_labels
    )
    
    if emit_debug:
        debug_eval("TERMINAL INTERVIEW OUTCOME", terminal_true_labels, terminal_aria_labels)
    
    return {
        "evaluation_type": "stored_belief_verdict",
        "evaluates_learned_policy": False,
        "num_episodes": len(episodes),
        "total_transitions": len(dataset),
        "rl_metrics": rl_metrics,
        "belief_verdict_metrics": terminal_response_metrics,
        # Backward-compatible alias for existing report consumers.
        "response_metrics": terminal_response_metrics,
        "overall_transitions_metrics": overall_response_metrics,
        "component_bootstrap_intervals": component_bootstrap_intervals(dataset),
    }


def component_bootstrap_intervals(dataset, samples=200, seed=42):
    """Bootstrap terminal metrics by independent resume/JD component."""
    components = connected_identity_components(dataset)
    if len(components) < 2:
        return {"available": False, "reason": "fewer_than_two_identity_components"}
    component_pairs = []
    for component in components:
        pairs = []
        for episode in component:
            if episode and episode[-1].get("true_label") in (0, 1, 2):
                pairs.append((episode[-1]["true_label"], episode[-1].get("aria_label")))
        component_pairs.append(pairs)
    rng = np.random.default_rng(seed)
    micro_values, macro_values, balanced_values, abstention_values = [], [], [], []
    for _ in range(samples):
        pairs = []
        for index in rng.integers(0, len(component_pairs), size=len(component_pairs)):
            pairs.extend(component_pairs[int(index)])
        if not pairs:
            continue
        metrics = compute_response_metrics(
            [prediction for _, prediction in pairs],
            [truth for truth, _ in pairs],
        )
        micro_values.append(metrics["overall_accuracy"])
        macro_values.append(metrics["macro_f1"])
        balanced_values.append(metrics["balanced_accuracy"])
        abstention_values.append(metrics["abstention_rate"])
    if not micro_values:
        return {"available": False, "reason": "no_terminal_pairs"}
    return {
        "available": True,
        "num_components": len(components),
        "samples": len(micro_values),
        "micro_f1_95_ci": np.percentile(micro_values, [2.5, 97.5]).tolist(),
        "macro_f1_95_ci": np.percentile(macro_values, [2.5, 97.5]).tolist(),
        "balanced_accuracy_95_ci": np.percentile(balanced_values, [2.5, 97.5]).tolist(),
        "abstention_rate_95_ci": np.percentile(abstention_values, [2.5, 97.5]).tolist(),
    }


def run_benchmark(dataset_file: str, output_file: str = "benchmark_report.json"):
    """Evaluate stored belief verdicts; this does not evaluate an IQL policy.

    Learned-policy evaluation requires loading a checkpoint and running fresh
    environment rollouts. Keeping that distinction explicit prevents belief
    classification metrics from being reported as policy performance.
    """
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_file}")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if not dataset:
        logger.error("Dataset is empty.")
        return

    report = build_belief_report(dataset, emit_debug=True)
    
    # Security fix: strip any path traversal from output_file
    safe_output_name = Path(output_file).name
    out_path = dataset_path.parent / safe_output_name
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n--- Stored Belief Verdict Report ({out_path.name}) ---")
    print("NOTE: This report does not load or evaluate the trained IQL policy.")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    ds_file = base_dir / "data" / "synthetic" / "qwen_rl_dataset.json"
    run_benchmark(str(ds_file))
