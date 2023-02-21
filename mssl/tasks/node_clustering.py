from typing import Dict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate_node_clustering(
    z: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Trains K-Means and computes metrics."""
    num_cls = y.unique().shape[0]

    metrics = {}

    for name, mask in (
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ):
        NMIs = []
        ARIs = []

        for i in range(10):
            kmeans = KMeans(
                n_clusters=num_cls,
                n_init="auto",
                random_state=seed + i,
            )
            kmeans.fit(z[mask])

            y_true = y[mask]
            y_pred = kmeans.predict(z[mask])

            NMIs.append(
                normalized_mutual_info_score(
                    labels_true=y_true,
                    labels_pred=y_pred,
                    average_method="arithmetic",
                )
            )

            ARIs.append(
                adjusted_rand_score(
                    labels_true=y_true,
                    labels_pred=y_pred,
                )
            )

        metrics[name] = {
            "NMI": np.mean(NMIs),
            "ARI": np.mean(ARIs),
        }

    return metrics
