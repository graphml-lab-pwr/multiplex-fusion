import torch
from torch_geometric.data import HeteroData

from mssl.tasks.node_classification import evaluate_node_classification
from mssl.tasks.node_clustering import evaluate_node_clustering
from mssl.tasks.similarity_search import evaluate_similarity_search


def evaluate_all_node_tasks(
    z: torch.Tensor,
    data: HeteroData,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    kwargs = dict(
        z=z,
        y=data["Node"].y.cpu(),
        train_mask=data["Node"].train_mask.cpu(),
        val_mask=data["Node"].val_mask.cpu(),
        test_mask=data["Node"].test_mask.cpu(),
        seed=seed,
    )

    metrics = {
        "classification": evaluate_node_classification(**kwargs),
        "clustering": evaluate_node_clustering(**kwargs),
        "similarity_search": evaluate_similarity_search(**kwargs),
    }

    return metrics
