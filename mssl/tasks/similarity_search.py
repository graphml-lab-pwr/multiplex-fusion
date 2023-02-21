from typing import Dict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_similarity_search(
    z: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """Trains K-Means and computes metrics."""
    metrics = {}

    for name, mask in (
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ):
        metrics[name] = {}

        num_nodes = z[mask].numpy().shape[0]

        cos_sim = cosine_similarity(z[mask].numpy()) - np.eye(num_nodes)

        for N in (5, 10, 20, 50):
            if N > num_nodes:
                continue

            indices = np.argsort(cos_sim, axis=1)[:, -N:]

            tmp = np.tile(y[mask].numpy(), (num_nodes, 1))

            selected_label = tmp[
                np.repeat(np.arange(num_nodes), N), indices.ravel()
            ].reshape(num_nodes, N)

            original_label = np.repeat(y[mask].numpy(), N).reshape(num_nodes, N)

            metrics[name][f"Sim@{N}"] = np.mean(
                np.sum((selected_label == original_label), 1) / N
            )

    return metrics
