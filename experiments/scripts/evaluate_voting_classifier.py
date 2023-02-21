"""Script for evaluating voting classifiers ("fusion on decision")."""
import json
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm

from mssl.utils import parse_seeds


def soft_voting(
    models: list[LogisticRegression],
    embeddings: list[torch.Tensor],
) -> np.ndarray:
    probas = np.asarray([
        clf.predict_proba(Z)
        for clf, Z in zip(models, embeddings)
    ])
    avg = np.average(probas, axis=0)
    maj = np.argmax(avg, axis=1)

    return maj


def hard_voting(
    models: list[LogisticRegression],
    embeddings: list[torch.Tensor],
) -> np.ndarray:
    predictions = np.asarray([
        clf.predict(Z)
        for clf, Z in zip(models, embeddings)
    ]).T
    maj = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)),
        axis=1,
        arr=predictions,
    )

    return maj


@click.command()
@click.option(
    "--seeds",
    help="List of seeds for PRNGs",
    type=str,
    callback=parse_seeds,
)
@click.option(
    "--dataset_file",
    help="Path to the graph dataset",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--input_dir",
    help="Path to directory with embeddings",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--soft_output_dir",
    help="Path to output directory for soft voting",
    type=click.Path(path_type=Path),
)
@click.option(
    "--hard_output_dir",
    help="Path to output directory for hard voting",
    type=click.Path(path_type=Path),
)
def main(
    seeds: list[int],
    dataset_file: str,
    input_dir: Path,
    soft_output_dir: Path,
    hard_output_dir: Path,
):
    """Trains models and saves embeddings."""
    soft_output_dir.mkdir(parents=True, exist_ok=True)
    hard_output_dir.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(iterable=seeds, desc="Seeds"):
        data = torch.load(dataset_file)

        train_mask = data["Node"].train_mask
        val_mask = data["Node"].val_mask
        test_mask = data["Node"].test_mask
        y = data["Node"].y

        layer_embeddings = [
            torch.load(f=input_dir / f"{layer_name}_{seed}.pt")
            for _, layer_name, _ in data.edge_types
        ]

        for voting_strategy, voting_fn, output_dir in (
            ("soft", soft_voting, soft_output_dir),
            ("hard", hard_voting, hard_output_dir),
        ):
            models = [
                LogisticRegression(random_state=seed, max_iter=1000)
                .fit(X=embedding[train_mask], y=y[train_mask])
                for embedding in layer_embeddings
            ]

            y_pred_full = voting_fn(models=models, embeddings=layer_embeddings)

            metrics = {
                "classification": {},
                "clustering": {
                    split: {"NMI": np.nan, "ARI": np.nan}
                    for split in ("train", "val", "test")
                },
                "similarity_search": {
                    split: {"Sim@5": np.nan}
                    for split in ("train", "val", "test")
                },
            }

            for name, mask in (
                ("train", train_mask),
                ("val", val_mask),
                ("test", test_mask),
            ):
                y_true = y[mask]
                y_pred = y_pred_full[mask]

                metrics["classification"][name] = classification_report(
                    y_true=y_true,
                    y_pred=y_pred,
                    output_dict=True,
                    zero_division=0,
                )

                metrics["classification"][name]["auc"] = np.nan

                metrics["classification"][name]["micro avg"] = {
                    "f1-score": f1_score(
                        y_true=y_true,
                        y_pred=y_pred,
                        average="micro",
                    ),
                }

            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with (metrics_dir / f"{seed}.json").open("w") as fout:
                json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
