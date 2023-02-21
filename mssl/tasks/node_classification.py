from typing import Dict

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score


def evaluate_node_classification(
    z: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Trains a logistic regression classifier and computes metrics."""

    clf = LogisticRegression(random_state=seed, max_iter=1_000)
    clf.fit(z[train_mask], y[train_mask])

    metrics = {}

    for name, mask in (
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ):
        y_true = y[mask]
        y_pred = clf.predict(z[mask])
        y_score = clf.predict_proba(z[mask])

        metrics[name] = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            zero_division=0,
        )

        metrics[name]["auc"] = roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            multi_class="ovr",
        )

        metrics[name]["micro avg"] = {
            "f1-score": f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        }

    return metrics
