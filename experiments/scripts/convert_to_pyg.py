"""Script for converting raw datasets into PyG format."""
import pickle
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import load_npz
from torch_geometric.data import HeteroData


def _build_node_info(raw) -> HeteroData:
    data = HeteroData()
    x = torch.from_numpy(raw["feature"]).float()
    x = x / x.sum(dim=-1, keepdim=True)
    data["Node"].x = x
    data["Node"].y = torch.from_numpy(raw["label"]).argmax(dim=-1)

    return data


def _add_split_masks(raw, data: HeteroData) -> HeteroData:
    data = data.clone()

    train_mask = torch.zeros(data["Node"].num_nodes)
    train_mask[raw["train_idx"]] = 1
    data["Node"].train_mask = train_mask.bool()

    val_mask = torch.zeros(data["Node"].num_nodes)
    val_mask[raw["val_idx"]] = 1
    data["Node"].val_mask = val_mask.bool()

    test_mask = torch.zeros(data["Node"].num_nodes)
    test_mask[raw["test_idx"]] = 1
    data["Node"].test_mask = test_mask.bool()

    return data


def _add_edges(raw, data: HeteroData, edge_type_names: List[str]) -> HeteroData:
    data = data.clone()
    for et in edge_type_names:
        data["Node", et, "Node"].edge_index = torch.stack(
            torch.from_numpy(raw[et]).nonzero(as_tuple=True)
        )

    return data


def convert_IMDB(input_dir: Path) -> HeteroData:
    with (input_dir / "IMDB.pkl").open("rb") as fin:
        raw = pickle.load(fin)

    data = _build_node_info(raw=raw)
    data = _add_split_masks(raw=raw, data=data)
    data = _add_edges(raw=raw, data=data, edge_type_names=["MAM", "MDM"])

    return data


def convert_Amazon(input_dir: Path) -> HeteroData:
    with (input_dir / "Amazon.pkl").open("rb") as fin:
        raw = pickle.load(fin)

    data = _build_node_info(raw=raw)
    data = _add_split_masks(raw=raw, data=data)
    data = _add_edges(
        raw=raw,
        data=data,
        edge_type_names=["IVI", "IBI", "IOI"],
    )

    return data


def convert_ACM(input_dir: Path) -> HeteroData:
    raw = loadmat(file_name=(input_dir / "ACM.mat").as_posix())

    data = _build_node_info(raw=raw)
    data = _add_split_masks(raw=raw, data=data)
    data = _add_edges(
        raw=raw,
        data=data,
        edge_type_names=["PLP", "PAP"],
    )

    return data


def convert_Freebase(input_dir: Path) -> HeteroData:
    data_dir = input_dir / "Freebase"

    data = HeteroData()

    data["Node"].y = torch.from_numpy(
        np.load(file=(data_dir / "labels.npy").as_posix())
    )
    data["Node"].x = torch.eye(data["Node"].y.shape[0])

    for metapath in ("mam", "mdm", "mwm"):
        adj = load_npz(file=(data_dir / f"{metapath}.npz").as_posix())

        data["Node", metapath.upper(), "Node"].edge_index = torch.stack(
            [
                torch.from_numpy(adj.row).long(),
                torch.from_numpy(adj.col).long(),
            ],
            dim=0,
        )

    split_idx = {
        "train_idx": np.load(file=(data_dir / "train_60.npy").as_posix()),
        "val_idx": np.load(file=(data_dir / "val_60.npy").as_posix()),
        "test_idx": np.load(file=(data_dir / "test_60.npy").as_posix()),
    }

    data = _add_split_masks(raw=split_idx, data=data)

    return data


@click.command()
@click.option(
    "--name",
    help="Name of the graph dataset",
    type=str,
)
@click.option(
    "--input_dir",
    help="Path to input directory of raw datasets",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output_dir",
    help="Path to output directory",
    type=click.Path(path_type=Path),
)
def main(name: str, input_dir: Path, output_dir: Path):
    """Extracts node features and saves them into a file."""
    if name == "IMDB":
        data = convert_IMDB(input_dir)
    elif name == "Amazon":
        data = convert_Amazon(input_dir)
    elif name == "ACM":
        data = convert_ACM(input_dir)
    elif name == "Freebase":
        data = convert_Freebase(input_dir)
    else:
        raise ValueError(f"Unknown dataset: '{name}'")

    output_dir.mkdir(exist_ok=True, parents=True)
    torch.save(obj=data, f=(output_dir / f"{name}.pt"))


if __name__ == "__main__":
    main()
