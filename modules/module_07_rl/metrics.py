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

def compute_rl_metrics(episodes_data):
    """Compute RL-specific metrics from a list of episodes."""
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
        "avg_episode_length": float(avg_length),
        "avg_cumulative_reward": float(np.mean(rewards)),
        "policy_entropy": float(np.mean(entropies))
    }

def compute_response_metrics(
    aria_labels,
    true_labels,
    questions=None,
    jd_text="",
    beliefs=None,
):
    """Compute Accuracy, F1, Kappa, Confusion Matrix, and optional ROUGE-L."""
    if not aria_labels or not true_labels:
        return {}
        
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
    micro_values, macro_values = [], []
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
        micro_values.append(metrics["micro_f1"])
        macro_values.append(metrics["macro_f1"])
    if not micro_values:
        return {"available": False, "reason": "no_terminal_pairs"}
    return {
        "available": True,
        "num_components": len(components),
        "samples": len(micro_values),
        "micro_f1_95_ci": np.percentile(micro_values, [2.5, 97.5]).tolist(),
        "macro_f1_95_ci": np.percentile(macro_values, [2.5, 97.5]).tolist(),
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
