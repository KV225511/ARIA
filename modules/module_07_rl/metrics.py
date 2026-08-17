import json
import os
import logging
from pathlib import Path
import numpy as np
from collections import Counter
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

def compute_response_metrics(aria_labels, true_labels, questions=None, jd_text=""):
    """Compute Accuracy, F1, Kappa, Confusion Matrix, and optional ROUGE-L."""
    if not aria_labels or not true_labels:
        return {}
        
    # Assuming labels are categorical (0: beginner, 1: mid, 2: expert)
    acc = accuracy_score(true_labels, aria_labels)
    macro_f1 = f1_score(true_labels, aria_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, aria_labels, average='micro', zero_division=0)
    precision = precision_score(true_labels, aria_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, aria_labels, average='macro', zero_division=0)
    kappa = cohen_kappa_score(true_labels, aria_labels)
    conf_mat = confusion_matrix(true_labels, aria_labels, labels=[0, 1, 2]).tolist()
    
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
        "cohens_kappa": float(kappa),
        "confusion_matrix": conf_mat,
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
        
    episodes = []
    current_ep = []
    
    # Terminal (final conclusion) labels
    terminal_aria_labels = []
    terminal_true_labels = []
    
    # All transitions labels
    all_aria_labels = []
    all_true_labels = []
    
    questions = []
    jd_texts = []
    
    for t in dataset:
        current_ep.append(t)
        
        # Collect overall labels
        if "true_label" in t and "aria_label" in t:
            all_true_labels.append(t["true_label"])
            all_aria_labels.append(t["aria_label"])
            
            # Collect terminal-only labels
            if t.get('done', False):
                terminal_true_labels.append(t["true_label"])
                terminal_aria_labels.append(t["aria_label"])
            
        if "question" in t and "jd_text" in t:
            questions.append(t["question"])
            if t["jd_text"] not in jd_texts:
                jd_texts.append(t["jd_text"])
                
        if t.get('done', False):
            episodes.append(current_ep)
            current_ep = []
            
    if current_ep:
        episodes.append(current_ep)
        # If last transition didn't have done=True, capture its terminal label
        if current_ep and "true_label" in current_ep[-1] and "aria_label" in current_ep[-1]:
            if not current_ep[-1].get('done', False):
                terminal_true_labels.append(current_ep[-1]["true_label"])
                terminal_aria_labels.append(current_ep[-1]["aria_label"])
        
    rl_metrics = compute_rl_metrics(episodes)
    
    jd_text_combined = " ".join(jd_texts)
    
    # Evaluate terminal decisions (honest assessment of interview outcome)
    terminal_response_metrics = compute_response_metrics(
        terminal_aria_labels, terminal_true_labels, questions, jd_text_combined
    )
    
    # Also record overall transitions response metrics
    overall_response_metrics = compute_response_metrics(
        all_aria_labels, all_true_labels
    )
    
    debug_eval("TERMINAL INTERVIEW OUTCOME", terminal_true_labels, terminal_aria_labels)
    
    report = {
        "evaluation_type": "stored_belief_verdict",
        "evaluates_learned_policy": False,
        "num_episodes": len(episodes),
        "total_transitions": len(dataset),
        "rl_metrics": rl_metrics,
        "belief_verdict_metrics": terminal_response_metrics,
        # Backward-compatible alias for existing report consumers.
        "response_metrics": terminal_response_metrics,
        "overall_transitions_metrics": overall_response_metrics
    }
    
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
