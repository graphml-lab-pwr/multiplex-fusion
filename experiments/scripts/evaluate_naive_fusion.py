"""Script for applying and evaluating naive fusion mechanisms."""
import json
from pathlib import Path

import click
import torch
from tqdm.auto import tqdm

from mssl.fusion import (
    ConcatFusion,
    MaxFusion,
    MeanFusion,
    MinFusion,
    SumFusion,
)
from mssl.tasks import evaluate_all_node_tasks
from mssl.utils import parse_seeds


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
    "--output_dir",
    help="Path to output directory",
    type=click.Path(path_type=Path),
)
def main(
    seeds: list[int],
    dataset_file: str,
    input_dir: Path,
    output_dir: Path,
):
    """Trains model and saves embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(iterable=seeds, desc="Seeds"):
        data = torch.load(dataset_file)

        edge_types = [et for _, et, _ in data.edge_types]
        layer_embeddings = {
            et: torch.load(f=input_dir / f"{et}_{seed}.pt")
            for et in edge_types
        }

        for fn_name, cls in tqdm(
            iterable=(
                ("Concat", ConcatFusion),
                ("Sum", SumFusion),
                ("Min", MinFusion),
                ("Mean", MeanFusion),
                ("Max", MaxFusion),
            ),
            desc="Naive fusion functions",
        ):
            z_fused = cls(edge_types)(zs=layer_embeddings)

            embeddings_dir = output_dir / fn_name / "embeddings"

            embeddings_dir.mkdir(parents=True, exist_ok=True)
            torch.save(obj=z_fused, f=embeddings_dir / f"{seed}.pt")

            metrics = evaluate_all_node_tasks(z=z_fused, data=data, seed=seed)

            metrics_dir = output_dir / fn_name / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with (metrics_dir / f"{seed}.json").open("w") as fout:
                json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
