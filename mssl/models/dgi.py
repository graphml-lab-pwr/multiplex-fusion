"""Implementation of DGI wrapper."""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import DeepGraphInfomax, GCN

from mssl.models.base import BaseGNN


class DGI(BaseGNN):
    """Implementation of DGI wrapper model."""

    def __init__(
        self,
        edge_type: str,
        num_layers: int,
        in_channels: int,
        emb_dim: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters()

        self.edge_type = edge_type

        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=emb_dim,
            num_layers=num_layers,
            act="relu",
        )
        self.dgi = DeepGraphInfomax(
            hidden_channels=emb_dim,
            encoder=self.encoder,
            summary=self.summary_fn,
            corruption=self.corruption_fn,
        )

        self.dgi.reset_parameters()

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "DGI":
        return cls(
            edge_type=hparams["edge_type"],
            in_channels=data["Node"].x.size(-1),
            emb_dim=hparams["emb_dim"],
            num_layers=hparams["num_layers"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    @staticmethod
    def summary_fn(z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.sigmoid(z.mean(dim=0))

    @staticmethod
    def corruption_fn(
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x[torch.randperm(x.size(0), device=x.device)], edge_index

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        return self.encoder(
            x=graph["Node"].x,
            edge_index=graph["Node", self.edge_type, "Node"].edge_index,
        )

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        pos_z, neg_z, summary = self.dgi(
            x=batch["Node"].x,
            edge_index=batch["Node", self.edge_type, "Node"].edge_index,
        )

        loss = self.dgi.loss(pos_z=pos_z, neg_z=neg_z, summary=summary)

        return loss


class FlattenedGraphDGI(DGI):

    @classmethod
    def from_hparams(
        cls,
        data: HeteroData,
        hparams: dict,
    ) -> "FlattenedGraphDGI":
        return cls(
            edge_type=None,  # Will not be used
            in_channels=data["Node"].x.size(-1),
            emb_dim=hparams["emb_dim"],
            num_layers=hparams["num_layers"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        return self.encoder(
            x=graph["Node"].x,
            edge_index=graph.to_homogeneous().edge_index,
        )

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        pos_z, neg_z, summary = self.dgi(
            x=batch["Node"].x,
            edge_index=batch.to_homogeneous().edge_index,
        )

        loss = self.dgi.loss(pos_z=pos_z, neg_z=neg_z, summary=summary)

        return loss
