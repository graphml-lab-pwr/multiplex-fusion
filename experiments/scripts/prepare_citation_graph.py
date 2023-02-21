"""Script for converting Cora/Citeseer to multiplex networks.

Source: https://www.frontiersin.org/articles/10.3389/fphy.2021.763904/full
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData
from torch_geometric.datasets import Planetoid


def prepare_citation_dataset(name: str) -> HeteroData:
    with TemporaryDirectory() as tmp_dir:
        data = Planetoid(
            root=tmp_dir,
            name=name,
        )[0]

    out = HeteroData()
    norm = data.x.sum(dim=-1, keepdim=True)
    norm[norm == 0] = 1
    out["Node"].x = data.x / norm
    out["Node"].y = data.y

    out["Node"].train_mask = data.train_mask
    out["Node"].val_mask = data.val_mask
    out["Node"].test_mask = data.test_mask

    out["Node", "Citation", "Node"].edge_index = data.edge_index

    k = 10
    knn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
    )
    knn.fit(data.x)
    neighbors = knn.kneighbors(X=data.x, return_distance=False)[:, 1:]

    src = torch.arange(data.num_nodes).repeat_interleave(k)
    dst = torch.from_numpy(neighbors.flatten())

    knn_edge_index = torch.cat([
        torch.stack([src, dst], dim=0),
        torch.stack([dst, src], dim=0),
    ], dim=-1)

    out["Node", "KNN", "Node"].edge_index = knn_edge_index

    return out


@click.command()
@click.option(
    "--name",
    help="Name of the graph dataset",
    type=str,
)
@click.option(
    "--output_dir",
    help="Path to output directory",
    type=click.Path(path_type=Path),
)
def main(name: str, output_dir: Path):
    data = prepare_citation_dataset(name=name)
    output_dir.mkdir(exist_ok=True, parents=True)
    torch.save(obj=data, f=(output_dir / f"{name}.pt"))


if __name__ == "__main__":
    main()
