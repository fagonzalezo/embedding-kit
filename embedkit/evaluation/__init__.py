"""Evaluation: metrics for scoring an embedding, and a protocol for scoring a transform."""

from embedkit.evaluation.metrics import (
    continuity,
    effective_rank,
    k_skewness,
    knn_accuracy,
    knn_overlap,
    linear_cka,
    linear_probe,
    nc1,
    recall_at_k,
    silhouette,
    trustworthiness,
)
from embedkit.evaluation.protocol import METRICS, compare, evaluate_transform

__all__ = [
    "METRICS",
    "compare",
    "evaluate_transform",
    "continuity",
    "effective_rank",
    "k_skewness",
    "knn_accuracy",
    "knn_overlap",
    "linear_cka",
    "linear_probe",
    "nc1",
    "recall_at_k",
    "silhouette",
    "trustworthiness",
]
