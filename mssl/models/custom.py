"""Implementation of a custom model."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, Sequential

from mssl.models.base import BaseGNN
from mssl.fusion import AttentionFusion, ConcatLinFusion
from mssl.utils import barlow_twins_loss


class F_GBT(BaseGNN):
    """Implementation of a custom self-supervised model."""

    def __init__(
        self,
        fusion_type: str,
        edge_types: list[str],
        feature_dim: int,
        mlp_dim: int,
        emb_dim: int,
        dropout: float,
        self_connection_weight: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters()

        self.edge_types = edge_types
        self.dropout = dropout
        self.self_connection_weight = self_connection_weight

        self.mlps = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(feature_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, emb_dim),
            )
            for edge_type in self.edge_types
        })

        self.gcns = nn.ModuleDict({
            edge_type: Sequential("x, edge_index, edge_weight", [
                (GCNConv(emb_dim, emb_dim), "x, edge_index, edge_weight -> x"),
                (nn.BatchNorm1d(emb_dim), "x -> x"),
                (nn.ReLU(), "x -> x"),
            ])
            for edge_type in self.edge_types
        })

        if fusion_type == "attention":
            self.fusion = AttentionFusion(
                emb_dim=emb_dim,
                edge_types=edge_types,
            )
        elif fusion_type == "concatlin":
            self.fusion = ConcatLinFusion(
                emb_dim=emb_dim,
                edge_types=edge_types,
            )
        else:
            raise ValueError(f"Unknown fusion type: '{fusion_type}'")

        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.mlps.values():
            for layer in mlp:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        for gcn in self.gcns.values():
            gcn.reset_parameters()

        self.fusion.reset_parameters()

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "F_GBT":
        return cls(
            fusion_type=hparams["fusion_type"],
            edge_types=[et for _, et, _ in data.edge_types],
            feature_dim=data["Node"].x.size(-1),
            mlp_dim=hparams["mlp_dim"],
            emb_dim=hparams["emb_dim"],
            dropout=hparams["dropout"],
            self_connection_weight=hparams["self_connection_weight"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(
        self,
        graph: HeteroData,
        return_layerwise_repr: bool = False,
    ) -> (
        torch.Tensor
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]
    ):
        hs, zs = {}, {}
        
        x = F.dropout(
            graph["Node"].x,
            p=self.dropout,
            training=self.training,
        )

        for edge_type in self.edge_types:
            edge_index = graph["Node", edge_type, "Node"].edge_index

            edge_weight = torch.ones(edge_index.shape[1], device=x.device)
            edge_weight[
                edge_index[0] == edge_index[1]
            ] = self.self_connection_weight

            hs[edge_type] = self.mlps[edge_type](x)
            zs[edge_type] = self.gcns[edge_type](
                x=hs[edge_type],
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        z_fused = self.fusion(zs)

        if return_layerwise_repr:
            return hs, zs, z_fused

        return z_fused

    def training_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        _, zs, z_fused = self.forward_repr(batch, return_layerwise_repr=True)

        loss = 0

        for et in self.edge_types:
            loss += barlow_twins_loss(zs[et], z_fused)

        return loss


class F_DGI(BaseGNN):

    def __init__(
        self,
        fusion_type: str,
        edge_types: list[str],
        feature_dim: int,
        emb_dim: int,
        dropout: float,
        self_connection_weight: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters()

        self.edge_types = edge_types
        self.dropout = dropout
        self.self_connection_weight = self_connection_weight

        self.gcns = nn.ModuleDict({
            edge_type: Sequential("x, edge_index, edge_weight", [
                (GCNConv(feature_dim, emb_dim), "x, edge_index, edge_weight -> x"),
                (nn.ReLU(), "x -> x"),
            ])
            for edge_type in self.edge_types
        })

        if fusion_type == "attention":
            self.fusion = AttentionFusion(
                emb_dim=emb_dim,
                edge_types=edge_types,
            )
        elif fusion_type == "concatlin":
            self.fusion = ConcatLinFusion(
                emb_dim=emb_dim,
                edge_types=edge_types,
            )
        else:
            raise ValueError(f"Unknown fusion type: '{fusion_type}'")

        self.discriminator = nn.Bilinear(emb_dim, emb_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for gcn in self.gcns.values():
            gcn.reset_parameters()

        self.fusion.reset_parameters()

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "F_DGI":
        return cls(
            fusion_type=hparams["fusion_type"],
            edge_types=[et for _, et, _ in data.edge_types],
            feature_dim=data["Node"].x.size(-1),
            emb_dim=hparams["emb_dim"],
            dropout=hparams["dropout"],
            self_connection_weight=hparams["self_connection_weight"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(
        self,
        graph: HeteroData,
        return_layerwise_repr: bool = False,
    ):
        x = graph["Node"].x

        zs_pos, zs_neg, zs_summaries = {}, {}, {}

        for edge_type in self.edge_types:
            edge_index = graph["Node", edge_type, "Node"].edge_index

            edge_weight = torch.ones(edge_index.shape[1], device=x.device)
            edge_weight[
                edge_index[0] == edge_index[1]
                ] = self.self_connection_weight

            # Positive view
            zs_pos[edge_type] = self.gcns[edge_type](
                x=F.dropout(x, p=self.dropout, training=self.training),
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

            # Negative view
            zs_neg[edge_type] = self.gcns[edge_type](
                x=F.dropout(x, p=self.dropout, training=self.training)[
                    torch.randperm(x.size(0), device=x.device),
                ],
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

            # Summary
            zs_summaries[edge_type] = (
                zs_pos[edge_type]
                .mean(dim=0, keepdim=True)
                .sigmoid()
            )

        z_fused_pos = self.fusion(zs_pos)
        z_fused_neg = self.fusion(zs_neg)
        z_fused_summary = z_fused_pos.mean(dim=0, keepdim=True).sigmoid()

        if return_layerwise_repr:
            return (
                zs_pos, zs_neg, zs_summaries,
                z_fused_pos, z_fused_neg, z_fused_summary,
            )

        return z_fused_pos

    def training_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        (
            zs_pos, zs_neg, zs_summaries,
            z_fused_pos, z_fused_neg, z_fused_summary,
        ) = self.forward_repr(batch, return_layerwise_repr=True)

        loss = 0.

        for edge_type in self.edge_types:
            loss += self._dgi_loss(
                pos=zs_pos[edge_type],
                neg=zs_neg[edge_type],
                summary=zs_summaries[edge_type],
            )

        loss += self._dgi_loss(
            pos=z_fused_pos,
            neg=z_fused_neg,
            summary=z_fused_summary,
        )

        return loss

    def _dgi_loss(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
        summary: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            input=torch.cat([
                self.discriminator(pos, summary.expand_as(pos)).squeeze(dim=-1),
                self.discriminator(neg, summary.expand_as(neg)).squeeze(dim=-1),
            ]),
            target=torch.cat([
                torch.ones(pos.shape[0]),
                torch.zeros(neg.shape[0]),
            ]).to(summary.device),
        )
