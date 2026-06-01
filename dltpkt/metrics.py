"""Evaluation metrics used by the DLTPKT Statics experiment."""

from __future__ import annotations

from typing import Dict

import torch


def binary_metrics(labels: torch.Tensor, scores: torch.Tensor) -> Dict[str, float]:
    """Compute AUC, accuracy, and the thresholded RMSE used in the paper code."""

    labels = labels.float().cpu().view(-1)
    scores = scores.float().cpu().view(-1)
    predictions = (scores >= 0.5).float()
    return {
        "auc": _roc_auc(labels, scores),
        "acc": float((predictions == labels).float().mean().item()),
        "rmse": float(torch.sqrt(torch.mean((labels - predictions) ** 2)).item()),
    }


def _roc_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Compute binary ROC AUC with average ranks for tied scores."""

    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both positive and negative examples")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)

    positive_rank_sum = 0.0
    start_rank = 1
    start_index = 0
    for count in counts.tolist():
        end_rank = start_rank + count - 1
        average_rank = (start_rank + end_rank) / 2.0
        positive_count = float(sorted_labels[start_index : start_index + count].sum().item())
        positive_rank_sum += average_rank * positive_count
        start_rank = end_rank + 1
        start_index += count

    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
