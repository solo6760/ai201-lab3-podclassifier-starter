import json
import os
from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples


def run_evaluation(labeled_examples: list[dict] = None, prompt_mode: str = "default", target_class: str = None) -> dict:
    """
    Run the classifier against the held-out test set and return full results.
    Allows passing custom labeled_examples, prompt_mode, and target_class for experiments.
    """
    if labeled_examples is None:
        labeled_examples = load_labeled_examples()

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    results = []
    for episode in test_episodes:
        print(f"  Classifying: {episode['title'][:60]}...")
        prediction = classify_episode(
            episode["description"], 
            labeled_examples, 
            prompt_mode=prompt_mode, 
            target_class=target_class
        )
        results.append({
            "id": episode["id"],
            "title": episode["title"],
            "description": episode["description"],
            "ground_truth": episode["label"],
            "predicted": prediction["label"],
            "confidence": prediction.get("confidence", 5),
            "reasoning": prediction["reasoning"],
            "correct": prediction["label"] == episode["label"],
        })

    predictions = [r["predicted"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]

    return {
        "results": results,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "total": len(results),
    }


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Compute overall classification accuracy.

    Accuracy = number of correct predictions / total predictions.
    A prediction is correct when it exactly matches the ground truth label.
    """
    if not predictions or not ground_truth:
        return 0.0
    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    return correct / len(predictions)


def compute_per_class_accuracy(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, dict]:
    """
    Compute accuracy broken down by each label class.

    For each label in VALID_LABELS, compute:
      - "correct"  : number of episodes with this ground-truth label predicted correctly
      - "total"    : number of episodes with this ground-truth label
      - "accuracy" : correct / total (0.0 if total is 0)

    Return a dict keyed by label. Example:
      {
        "interview": {"correct": 4, "total": 5, "accuracy": 0.8},
        "solo":      {"correct": 5, "total": 5, "accuracy": 1.0},
        ...
      }
    """
    results = {label: {"correct": 0, "total": 0, "accuracy": 0.0} for label in VALID_LABELS}
    for p, gt in zip(predictions, ground_truth):
        if gt in results:
            results[gt]["total"] += 1
            if p == gt:
                results[gt]["correct"] += 1

    for label in VALID_LABELS:
        stats = results[label]
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0.0

    return results


def format_evaluation_report(eval_results: dict) -> str:
    """
    Format evaluation results into a readable report string.

    This function is already complete. Pass it the dict returned by run_evaluation().
    """
    predictions = eval_results["predictions"]
    ground_truth = eval_results["ground_truth"]
    results = eval_results["results"]

    accuracy = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)

    # Calculate average confidence per class
    confidences = {label: [] for label in VALID_LABELS}
    for r in results:
        gt = r["ground_truth"]
        if gt in confidences:
            confidences[gt].append(r["confidence"])

    avg_confidences = {}
    for label in VALID_LABELS:
        scores = confidences[label]
        avg_confidences[label] = sum(scores) / len(scores) if scores else 0.0

    lines = [
        f"## Evaluation Results\n",
        f"**Overall accuracy:** {accuracy:.1%} ({sum(r['correct'] for r in results)}/{eval_results['total']})\n",
        "\n**Per-class accuracy:**",
    ]
    for label, stats in per_class.items():
        bar = "█" * int(stats["accuracy"] * 10) + "░" * (10 - int(stats["accuracy"] * 10))
        lines.append(f"  {label:<12} {bar}  {stats['accuracy']:.0%}  ({stats['correct']}/{stats['total']}) [Avg Conf: {avg_confidences[label]:.1f}/10]")

    correct_confs = [r["confidence"] for r in results if r["correct"]]
    incorrect_confs = [r["confidence"] for r in results if not r["correct"]]
    avg_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    avg_incorrect = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0

    lines.append(f"\n**Confidence analysis:**")
    lines.append(f"  Average confidence for correct classifications: {avg_correct:.1f}/10")
    lines.append(f"  Average confidence for incorrect classifications: {avg_incorrect:.1f}/10")

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        lines.append(f"\n**Misclassified ({len(misclassified)}):**")
        for r in misclassified:
            lines.append(f"  [{r['ground_truth']} → {r['predicted']}] {r['title']} (Confidence: {r['confidence']})")
    else:
        lines.append("\n**No misclassifications — perfect score!**")

    return "\n".join(lines)
