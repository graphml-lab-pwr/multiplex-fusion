"""Implementation of S^2MGRL model."""
from itertools import combinations

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, Sequential
from tqdm.auto import tqdm

from mssl.models.base import BaseGNN
from mssl.fusion import AttentionFusion
from mssl.utils import barlow_twins_loss


class S2MGRL(BaseGNN):
    """Implementation of Simple Self-supervised Multiplex Graph
    Representation Learning model."""

    def __init__(
        self,
        edge_types: list[str],
        feature_dim: int,
        semantic_dim: int,
        emb_dim: int,
        dropout: float,
        self_connection_weight: float,
        omega_intra: dict[str, float],
        omega_inter: dict[str, float],
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)

        self.save_hyperparameters()

        self.edge_types = edge_types

        self.mlps = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(feature_dim, semantic_dim),
                nn.ReLU(),
                nn.Linear(semantic_dim, emb_dim),
            )
            for edge_type in self.edge_types
        })
        self.gcns = nn.ModuleDict({
            edge_type: Sequential("x, edge_index, edge_weight", [
                (GCNConv(emb_dim, emb_dim), "x, edge_index, edge_weight -> x"),
                (nn.ReLU(), "x -> x"),
            ])
            for edge_type in self.edge_types
        })

        self.dropout = dropout
        self.self_connection_weight = self_connection_weight
        self.omega_intra = omega_intra
        self.omega_inter = omega_inter

        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.mlps.values():
            for layer in mlp:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        for gcn in self.gcns.values():
            gcn.reset_parameters()

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "S2MGRL":
        return cls(
            edge_types=[et for _, et, _ in data.edge_types],
            feature_dim=data["Node"].x.size(-1),
            semantic_dim=hparams["semantic_dim"],
            emb_dim=hparams["emb_dim"],
            dropout=hparams["dropout"],
            self_connection_weight=hparams["self_connection_weight"],
            omega_intra=hparams["omega_intra"],
            omega_inter=hparams["omega_inter"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(
        self,
        graph: HeteroData,
        return_h: bool = False,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        hs = {}
        zs = {}

        x = F.dropout(
            input=graph["Node"].x,
            p=self.dropout,
            training=self.training,
        )

        for edge_type in self.edge_types:
            h = self.mlps[edge_type](x)

            edge_index = graph["Node", edge_type, "Node"].edge_index
            edge_weight = torch.ones(edge_index.shape[1], device=h.device)
            edge_weight[
                edge_index[0] == edge_index[1]
            ] = self.self_connection_weight

            z = self.gcns[edge_type](
                x=h,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

            hs[edge_type] = h
            zs[edge_type] = z

        if return_h:
            return hs, zs

        return zs

    def training_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        hs, zs = self.forward_repr(graph=batch, return_h=True)

        # Intra loss
        loss_intra = 0

        for edge_type in self.edge_types:
            loss_intra += (
                self.omega_intra[edge_type]
                * barlow_twins_loss(hs[edge_type], zs[edge_type])
            )

        # Inter loss
        loss_inter = 0

        for et1, et2 in combinations(self.edge_types, r=2):
            loss_inter += (
                self.omega_inter[f"{et1}_{et2}"]
                * barlow_twins_loss(zs[et1], zs[et2])
            )

        # Total loss
        loss = loss_intra + loss_inter

        return loss

    def validation_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ):
        """Train combination model (attention)."""
        z = train_fusion(
            zs=self.forward_repr(batch),
            y=batch["Node"].y,
            edge_types=self.edge_types,
            train_mask=batch["Node"].train_mask,
            val_mask=batch["Node"].val_mask,
            device=self.device,
        ).cpu()
        data = batch.cpu()

        return z, data

    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        z = train_fusion(
            zs=self.forward_repr(batch),
            y=batch["Node"].y,
            edge_types=self.edge_types,
            train_mask=batch["Node"].train_mask,
            val_mask=batch["Node"].val_mask,
            device=self.device,
        ).cpu()
        return z

    def configure_optimizers(self):
        """Prepares the optimizer module."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


@torch.enable_grad()
def train_fusion(
    zs: dict[str, torch.Tensor],
    y: torch.Tensor,
    edge_types: list[str],
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
    num_epochs: int = 100,
    check_val_every_n_epoch: int = 5,
) -> torch.Tensor:
    emb_dim = list(zs.values())[0].shape[-1]
    num_cls = y.unique().shape[0]

    fusion = AttentionFusion(
        emb_dim=emb_dim,
        edge_types=edge_types,
    ).to(device)
    lr = nn.Linear(emb_dim, num_cls).to(device)

    optimizer = torch.optim.AdamW(
        params=list(fusion.parameters()) + list(lr.parameters()),
        lr=1e-3,
        weight_decay=5e-4,
    )

    best_val_macro_f1 = -1
    best_z = None

    for epoch in tqdm(
        iterable=range(1, num_epochs + 1),
        desc="Train fusion model",
        leave=False,
    ):
        optimizer.zero_grad()

        loss = F.cross_entropy(
            input=lr(fusion(zs))[train_mask],
            target=y[train_mask],
        )

        loss.backward()
        optimizer.step()

        if epoch % check_val_every_n_epoch == 0:
            z = fusion(zs)
            f1_macro = f1_score(
                y_true=y[val_mask].cpu(),
                y_pred=lr(z)[val_mask].argmax(dim=-1).cpu(),
                average="macro",
            )

            if f1_macro > best_val_macro_f1:
                best_val_macro_f1 = f1_macro
                best_z = z.detach()

    return best_z
