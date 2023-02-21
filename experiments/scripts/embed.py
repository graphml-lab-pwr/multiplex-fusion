"""Script for computing and evaluating embeddings."""
import importlib
import json
import os
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import seed_everything
from torch_geometric.data.lightning import LightningNodeData
from tqdm.auto import tqdm

from mssl.models import DeepWalkDataModule
from mssl.lightning import train
from mssl.tasks import evaluate_all_node_tasks
from mssl.utils import parse_seeds


def get_model_cls(path: str) -> pl.LightningModule:
    package, cls_name = path.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(package), cls_name)


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
    "--config_file",
    help="Path to YAML config file",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--output_dir",
    help="Path to output directory",
    type=click.Path(path_type=Path),
)
def main(
    seeds: list[int],
    dataset_file: Path,
    config_file: Path,
    output_dir: Path,
):
    """Trains model and saves embeddings."""
    for seed in tqdm(iterable=seeds, desc="Seeds"):
        seed_everything(seed, workers=True)

        with config_file.open("r") as fin:
            hparams = yaml.safe_load(fin)[dataset_file.stem]

        hparams["seed"] = seed
        model_cls = hparams["model_cls"]

        data = torch.load(dataset_file)

        model = get_model_cls(model_cls).from_hparams(
            data=data,
            hparams=hparams,
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

        embedding = train(
            model=model,
            datamodule=datamodule,
            checkpoint_dir=output_dir / "models",
            logs_dir=output_dir / "logs" / str(seed),
            hparams=hparams,
        )

        (output_dir / "embeddings").mkdir(exist_ok=True, parents=True)
        torch.save(obj=embedding, f=output_dir / "embeddings" / f"{seed}.pt")

        metrics = evaluate_all_node_tasks(
            z=embedding,
            data=data,
            seed=seed,
        )

        (output_dir / "metrics").mkdir(exist_ok=True, parents=True)
        with (output_dir / "metrics" / f"{seed}.json").open("w") as fout:
            json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
