"""
Offline tuning utilities for centroid-based guardrail

This module provides tools for:
1. Threshold tuning based on development/validation data
2. Analyzing near-miss rejections
3. Adding new paraphrases from near-misses
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

from src.guardrails.centroid_guardrail import CentroidGuardrail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dev_data(dev_file: str) -> List[Tuple[str, str]]:
    """
    Load development/validation data for tuning

    Expected format: JSONL file with each line containing:
    {"text": "user query", "intent": "intent_name"} or {"text": "...", "intent": "none"}

    Args:
        dev_file: Path to development data file

    Returns:
        List of (text, gold_intent) tuples
    """
    data = []

    with open(dev_file, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append((item['text'], item['intent']))

    logger.info(f"Loaded {len(data)} examples from {dev_file}")
    return data


def evaluate_thresholds(
    guardrail: CentroidGuardrail,
    dev_data: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Tune per-intent thresholds using F1 score optimization

    For each intent, sweeps threshold values and finds the one with best F1.

    Args:
        guardrail: Initialized CentroidGuardrail instance
        dev_data: List of (text, gold_intent) tuples

    Returns:
        Dictionary mapping intent to optimal threshold
    """
    logger.info("Evaluating thresholds on development data...")

    # Collect predictions for all examples
    predictions = []
    for text, gold_intent in dev_data:
        best_intent, score, margin, all_scores = guardrail.score_intents(text)
        predictions.append((text, gold_intent, best_intent, score, all_scores))

    # Group by predicted intent
    by_intent = {}
    for text, gold, pred, score, all_scores in predictions:
        by_intent.setdefault(pred, []).append((gold, score))

    # Tune threshold for each intent
    optimal_thresholds = {}

    for intent, rows in by_intent.items():
        best_f1 = 0.0
        best_threshold = 0.70

        # Sweep threshold values
        for threshold in np.linspace(0.55, 0.85, 31):
            # Calculate metrics at this threshold
            tp = sum(1 for gold, score in rows if gold == intent and score >= threshold)
            fp = sum(1 for gold, score in rows if gold != intent and score >= threshold)
            fn = sum(1 for gold, score in rows if gold == intent and score < threshold)

            # Calculate precision, recall, F1
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds[intent] = float(best_threshold)

        logger.info(
            f"Intent '{intent}': optimal threshold={best_threshold:.3f}, F1={best_f1:.3f}"
        )

    return optimal_thresholds


def analyze_near_misses(log_file: str, top_n: int = 20) -> List[Dict]:
    """
    Analyze near-miss rejections from logs

    Near-misses are queries that were close to an intent but rejected.
    These are candidates for adding as new paraphrases.

    Args:
        log_file: Path to near-miss log file (JSONL format)
        top_n: Number of top near-misses to return per intent

    Returns:
        List of near-miss dictionaries, sorted by score (descending)
    """
    if not Path(log_file).exists():
        logger.warning(f"Near-miss log file not found: {log_file}")
        return []

    # Load all near-misses
    near_misses = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'near_miss':
                        near_misses.append(entry)
                except json.JSONDecodeError:
                    continue

    # Group by intent
    by_intent = {}
    for entry in near_misses:
        intent = entry.get('intent')
        if intent:
            by_intent.setdefault(intent, []).append(entry)

    # Get top N per intent
    top_misses = []
    for intent, entries in by_intent.items():
        # Sort by score (descending)
        sorted_entries = sorted(entries, key=lambda x: x.get('score', 0), reverse=True)
        top_misses.extend(sorted_entries[:top_n])

    logger.info(f"Found {len(near_misses)} total near-misses")
    logger.info(f"Returning top {top_n} per intent: {len(top_misses)} total")

    return top_misses


def add_paraphrases_from_near_misses(
    guardrail: CentroidGuardrail,
    near_misses: List[Dict],
    min_score: float = 0.60,
    max_per_intent: int = 5
) -> Dict[str, List[str]]:
    """
    Add high-quality near-misses as new paraphrases

    Args:
        guardrail: CentroidGuardrail instance to update
        near_misses: List of near-miss dictionaries
        min_score: Minimum score to consider adding
        max_per_intent: Maximum paraphrases to add per intent

    Returns:
        Dictionary mapping intent to list of added paraphrases
    """
    added = {}

    # Group by intent
    by_intent = {}
    for entry in near_misses:
        intent = entry.get('intent')
        score = entry.get('score', 0)

        if intent and score >= min_score:
            by_intent.setdefault(intent, []).append(entry)

    # Add top paraphrases per intent
    for intent, entries in by_intent.items():
        # Sort by score
        sorted_entries = sorted(entries, key=lambda x: x.get('score', 0), reverse=True)

        # Add up to max_per_intent
        added[intent] = []
        for entry in sorted_entries[:max_per_intent]:
            text = entry.get('user_text')
            if text:
                try:
                    guardrail.add_paraphrase(intent, text, rebuild=False)
                    added[intent].append(text)
                    logger.info(f"Added paraphrase to '{intent}': {text}")
                except Exception as e:
                    logger.warning(f"Failed to add paraphrase: {e}")

    # Rebuild all centroids once at the end
    if added:
        logger.info("Rebuilding centroids...")
        guardrail._build_centroids()

    return added


def generate_evaluation_report(
    guardrail: CentroidGuardrail,
    dev_data: List[Tuple[str, str]],
    output_file: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive evaluation report

    Args:
        guardrail: CentroidGuardrail instance
        dev_data: Development data for evaluation
        output_file: Optional file to save report

    Returns:
        Report dictionary
    """
    logger.info("Generating evaluation report...")

    # Evaluate on dev data
    correct = 0
    total = 0
    by_intent = {}

    for text, gold_intent in dev_data:
        result = guardrail.decide(text)
        predicted_intent = result.get('intent')

        total += 1
        if predicted_intent == gold_intent:
            correct += 1

        # Track per-intent accuracy
        by_intent.setdefault(gold_intent, {'correct': 0, 'total': 0})
        by_intent[gold_intent]['total'] += 1
        if predicted_intent == gold_intent:
            by_intent[gold_intent]['correct'] += 1

    # Calculate metrics
    overall_accuracy = correct / total if total > 0 else 0

    per_intent_accuracy = {
        intent: stats['correct'] / stats['total']
        for intent, stats in by_intent.items()
    }

    report = {
        "overall_accuracy": overall_accuracy,
        "total_examples": total,
        "correct_predictions": correct,
        "per_intent_accuracy": per_intent_accuracy,
        "intent_info": guardrail.get_intent_info()
    }

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.2%}")
    logger.info(f"Total Examples: {total}")
    logger.info(f"Correct Predictions: {correct}")
    logger.info(f"\nPer-Intent Accuracy:")
    for intent, acc in per_intent_accuracy.items():
        logger.info(f"  {intent}: {acc:.2%}")
    logger.info(f"{'='*60}\n")

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune guardrail thresholds")
    parser.add_argument(
        "--dev-file",
        type=str,
        default="data/dev_data.jsonl",
        help="Development data file (JSONL format)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/intent_paraphrases.json",
        help="Intent paraphrases configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/tuned_thresholds.json",
        help="Output file for tuned thresholds"
    )
    parser.add_argument(
        "--near-misses",
        type=str,
        default="logs/near_misses.jsonl",
        help="Near-miss log file to analyze"
    )
    parser.add_argument(
        "--add-paraphrases",
        action="store_true",
        help="Add high-quality near-misses as paraphrases"
    )

    args = parser.parse_args()

    # Initialize guardrail
    logger.info("Initializing guardrail...")
    guardrail = CentroidGuardrail(intent_paraphrases_path=args.config)

    # Load dev data if exists
    if Path(args.dev_file).exists():
        dev_data = load_dev_data(args.dev_file)

        # Tune thresholds
        optimal_thresholds = evaluate_thresholds(guardrail, dev_data)

        # Save tuned thresholds
        with open(args.output, 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        logger.info(f"Saved tuned thresholds to {args.output}")

        # Generate evaluation report
        generate_evaluation_report(
            guardrail,
            dev_data,
            output_file="reports/evaluation_report.json"
        )
    else:
        logger.warning(f"Dev file not found: {args.dev_file}")

    # Analyze near-misses
    if Path(args.near_misses).exists():
        near_misses = analyze_near_misses(args.near_misses)

        logger.info(f"\nTop Near-Misses:")
        for entry in near_misses[:10]:
            logger.info(
                f"  Intent: {entry['intent']} | "
                f"Score: {entry['score']:.3f} | "
                f"Text: {entry['user_text']}"
            )

        # Optionally add as paraphrases
        if args.add_paraphrases and near_misses:
            added = add_paraphrases_from_near_misses(guardrail, near_misses)

            logger.info(f"\nAdded paraphrases:")
            for intent, phrases in added.items():
                logger.info(f"  {intent}: {len(phrases)} paraphrases")

            # Save updated configuration
            output_config = args.config.replace('.json', '_updated.json')
            guardrail.save_config(output_config)
            logger.info(f"Saved updated configuration to {output_config}")
    else:
        logger.info(f"No near-miss log found at {args.near_misses}")
