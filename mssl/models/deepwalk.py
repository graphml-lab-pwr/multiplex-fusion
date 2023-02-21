"""Implementation of DeepWalk wrapper."""
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.nn import Node2Vec

from mssl.models.base import BaseGNN


class DeepWalk(BaseGNN):
    """Implementation of DeepWalk wrapper model."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        emb_dim: int,
        walk_length: int,
        context_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters(ignore=["edge_index"])

        self.deepwalk = Node2Vec(
            edge_index=edge_index,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            p=1,
            q=1,
            num_nodes=num_nodes,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.deepwalk.reset_parameters()

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "DeepWalk":
        return cls(
            edge_index=data["Node", hparams["edge_type"], "Node"].edge_index,
            num_nodes=data["Node"].num_nodes,
            emb_dim=hparams["emb_dim"],
            walk_length=hparams["walk_length"],
            context_size=hparams["context_size"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        return self.deepwalk().detach()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        pos_rw, neg_rw = batch
        return self.deepwalk.loss(pos_rw=pos_rw, neg_rw=neg_rw)


class DeepWalkDataModule(LightningNodeData):

    def __init__(self, data: HeteroData, train_loader: DataLoader):
        super().__init__(data=data, loader="full")

        self._train_dataloader = train_loader

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader


class FlattenedGraphDeepWalk(DeepWalk):

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "DeepWalk":
        return cls(
            edge_index=data.to_homogeneous().edge_index,
            num_nodes=data["Node"].num_nodes,
            emb_dim=hparams["emb_dim"],
            walk_length=hparams["walk_length"],
            context_size=hparams["context_size"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )
