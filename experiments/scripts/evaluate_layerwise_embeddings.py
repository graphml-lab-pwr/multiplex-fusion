"""Script for evaluating pre-computed layerwise embeddings."""
import json
from pathlib import Path

import click
import torch
from tqdm.auto import tqdm

from mssl.tasks import evaluate_all_node_tasks
from mssl.utils import parse_seeds


@click.command()
@click.option(
    "--seeds",
    help="List of seed for PRNGs",
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
    dataset_file: Path,
    input_dir: Path,
    output_dir: Path,
):
    """Trains model and saves embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(iterable=seeds, desc="Seeds"):
        data = torch.load(dataset_file)

        for path in tqdm(
            iterable=list(input_dir.glob(f"*_{seed}.pt")),
            desc="Evaluate layerwise embeddings",
        ):
            layer_name = path.stem.split("_")[0]

            if layer_name == "full":
                continue

            current_output_dir = output_dir / layer_name / "metrics"
            current_output_dir.mkdir(parents=True, exist_ok=True)

            z = torch.load(path)

            metrics = evaluate_all_node_tasks(z=z, data=data, seed=seed)

            with (current_output_dir / f"{seed}.json").open("w") as fout:
                json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
