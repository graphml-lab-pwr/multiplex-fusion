"""Script for extracting and evaluating node features."""
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
    "--output_dir",
    help="Path to output directory",
    type=click.Path(path_type=Path),
)
def main(seeds: list[int], dataset_file: Path, output_dir: Path):
    """Extracts node features and saves them into a file."""
    data = torch.load(dataset_file)

    embeddings = data["Node"].x

    output_dir.mkdir(exist_ok=True, parents=True)

    torch.save(obj=embeddings, f=output_dir / "embedding.pt")

    (output_dir / "metrics").mkdir(exist_ok=True, parents=True)

    for seed in tqdm(iterable=seeds, desc="Seeds"):
        metrics = evaluate_all_node_tasks(
            z=embeddings,
            data=data,
            seed=seed,
        )

        with (output_dir / "metrics" / f"{seed}.json").open("w") as fout:
            json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
