"""Script for running HPS grid search."""
import importlib
import json
import os
from itertools import combinations, product
from pathlib import Path

import click
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import seed_everything
from torch_geometric.data.lightning import LightningNodeData
from tqdm.auto import tqdm

from mssl.models import DeepWalkDataModule
from mssl.lightning import train_silent_without_validation
from mssl.tasks import evaluate_node_classification


def make_hyperparameter_search_grid(grid_values: dict) -> list[dict]:
    names = grid_values.keys()
    return [
        dict(zip(names, values))
        for values in product(*[grid_values[name] for name in names])
    ]


def make_hyperparameter_search_grid_s2mgrl(
    edge_types: list[str],
    grid_values: dict,
) -> list[dict]:
    # For each edge type
    omega_intra = [
        dict(zip(edge_types, oi))
        for oi in product(grid_values["omega_intra"], repeat=len(edge_types))
    ]

    # For each pair of edge types
    et_pairs = [f"{et1}_{et2}" for et1, et2 in combinations(edge_types, r=2)]

    omega_inter = [
        dict(zip(et_pairs, oi))
        for oi in product(grid_values["omega_inter"], repeat=len(et_pairs))
    ]

    return [
        dict(zip(["omega_intra", "omega_inter"], values))
        for values in product(omega_intra, omega_inter)
    ]


def get_model_cls(path: str) -> pl.LightningModule:
    package, cls_name = path.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(package), cls_name)


@click.command()
@click.option(
    "--seed",
    help="Seed for PRNGs",
    type=int,
)
@click.option(
    "--dataset_file",
    help="Path to the graph dataset",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--config_file",
    help="Path to YAML config file",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--log_file",
    help="Path to output log file (CSV file)",
    type=click.Path(path_type=Path),
)
@click.option(
    "--num_epochs",
    help="Numer of epochs to train model",
    type=int,
)
def main(
    seed: int,
    dataset_file: Path,
    config_file: Path,
    log_file: Path,
    num_epochs: int,
):
    """Evaluates a given model with different hyperparameters."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with config_file.open("r") as fin:
        hparams = yaml.safe_load(fin)[dataset_file.stem]

    data = torch.load(dataset_file)

    if "s2mgrl" in config_file.as_posix():
        grid = make_hyperparameter_search_grid_s2mgrl(
            edge_types=[et for _, et, _ in data.edge_types],
            grid_values=hparams["grid"],
        )
    else:
        grid = make_hyperparameter_search_grid(grid_values=hparams["grid"])

    log = []

    for current_hparams in tqdm(iterable=grid, desc="Hyperparameter grid"):
        # Validate hyperparameters
        if all(
            name in current_hparams
            for name in ("context_size", "walk_length")
        ):
            if current_hparams["context_size"] > current_hparams["walk_length"]:
                continue

        seed_everything(seed, workers=True)

        model_cls = get_model_cls(hparams["constants"]["model_cls"])

        model = model_cls.from_hparams(
            data=data,
            hparams={
                "seed": seed,
                **current_hparams,
                **hparams["constants"],
            },
        )

        if "deepwalk" in config_file.as_posix():
            datamodule = DeepWalkDataModule(
                data=data,
                train_loader=model.deepwalk.loader(
                    batch_size=data["Node"].num_nodes,
                    shuffle=True,
                    num_workers=int(os.getenv("NUM_WORKERS", 0)),
                ),
            )
        else:
            datamodule = LightningNodeData(data=data, loader="full")

        embeddings = train_silent_without_validation(
            model=model,
            datamodule=datamodule,
            num_epochs=num_epochs,
        )

        if "s2mgrl" in config_file.as_posix():
            current_hparams = {
                "omega_intra": json.dumps(current_hparams["omega_intra"]),
                "omega_inter": json.dumps(current_hparams["omega_inter"]),
            }

        metrics = evaluate_node_classification(
            z=embeddings,
            y=data["Node"].y.cpu(),
            train_mask=data["Node"].train_mask.cpu(),
            val_mask=data["Node"].val_mask.cpu(),
            test_mask=data["Node"].test_mask.cpu(),
            seed=seed,
        )

        metrics_to_log = {
            "macro_f1": metrics["val"]["macro avg"]["f1-score"],
            "micro_f1": metrics["val"]["micro avg"]["f1-score"],
            "auc": metrics["val"]["auc"],
            "accuracy": metrics["val"]["accuracy"],
        }

        log.append({**current_hparams, **metrics_to_log})

        pd.DataFrame.from_records(log).to_csv(path_or_buf=log_file, index=False)


if __name__ == "__main__":
    main()
