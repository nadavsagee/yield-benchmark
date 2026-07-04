"""Scorer — compare agent findings against ground truth."""

from scorer.score import (
    aggregate_scores,
    load_ground_truth,
    score_batch,
    score_dataset,
)

__all__ = [
    "aggregate_scores",
    "load_ground_truth",
    "score_batch",
    "score_dataset",
]
