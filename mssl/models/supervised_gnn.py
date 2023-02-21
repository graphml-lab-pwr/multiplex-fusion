"""Implementation of supervised GNN models."""
from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCN, GAT

from mssl.models.base import BaseGNN


class SupervisedGNN(BaseGNN):
    """Implementation of a supervised GNN model."""

    def __init__(
        self,
        edge_type: str,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters()

        self.edge_type = edge_type

        self.gnn = self.get_gnn(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            act="relu",
        )
        self.head = nn.Linear(hidden_channels, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "SupervisedGNN":
        return cls(
            edge_type=hparams["edge_type"],
            in_channels=data["Node"].x.shape[-1],
            hidden_channels=hparams["emb_dim"],
            num_classes=data["Node"].y.unique().shape[0],
            num_layers=hparams["num_layers"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    @abstractmethod
    def get_gnn(self, **kwargs):
        pass

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        return self.gnn(
            x=graph["Node"].x,
            edge_index=graph["Node", self.edge_type, "Node"].edge_index,
        )

    def training_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        mask = batch["Node"].train_mask

        y_true = batch["Node"].y[mask]
        y_pred = self.head(self.forward_repr(batch))[mask]

        return F.cross_entropy(input=y_pred, target=y_true)


class SupervisedGCN(SupervisedGNN):
    """Implementation of a supervised GCN model."""

    def get_gnn(self, **kwargs):
        return GCN(**kwargs)


class SupervisedGAT(SupervisedGNN):
    """Implementation of a supervised GAT model."""

    def get_gnn(self, **kwargs):
        return GAT(**kwargs)


class FlattenedGraphSupervisedGNN(SupervisedGNN):

    @classmethod
    def from_hparams(
        cls,
        data: HeteroData,
        hparams: dict,
    ) -> "FlattenedGraphSupervisedGNN":
        return cls(
            edge_type=None,  # Will not be used
            in_channels=data["Node"].x.shape[-1],
            hidden_channels=hparams["emb_dim"],
            num_classes=data["Node"].y.unique().shape[0],
            num_layers=hparams["num_layers"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        return self.gnn(
            x=graph["Node"].x,
            edge_index=graph.to_homogeneous().edge_index,
        )


class FlattenedGraphSupervisedGCN(FlattenedGraphSupervisedGNN):

    def get_gnn(self, **kwargs):
        return GCN(**kwargs)


class FlattenedGraphSupervisedGAT(FlattenedGraphSupervisedGNN):

    def get_gnn(self, **kwargs):
        return GAT(**kwargs)
