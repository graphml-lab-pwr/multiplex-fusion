"""Implementation DMGI model."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv

from mssl.models.base import BaseGNN


class DMGI(BaseGNN):
    """Implementation based on:
    `https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/dmgi_unsup.py`
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        alpha: float,
        lr: float,
        weight_decay: float,
        seed: int,
        w: float = 3.0,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters()

        self.convs = nn.ModuleList([
            GCNConv(in_channels, out_channels)
            for _ in range(num_relations)
        ])
        self.M = nn.Bilinear(out_channels, out_channels, 1)
        self.Z = nn.Parameter(torch.Tensor(num_nodes, out_channels))
        self.reset_parameters()

        self.alpha = alpha
        self.w = w

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "DMGI":
        return cls(
            num_nodes=data["Node"].num_nodes,
            in_channels=data["Node"].x.size(-1),
            out_channels=hparams["emb_dim"],
            num_relations=len(data.edge_types),
            alpha=hparams["alpha"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        return self.Z.data

    def training_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        x = batch["Node"].x
        edge_indices = batch.edge_index_dict.values()

        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)
            edge_weight[edge_index[0] == edge_index[1]] = self.w

            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(
                x=pos_h,
                edge_index=edge_index,
                edge_weight=edge_weight,
            ).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(
                x=neg_h,
                edge_index=edge_index,
                edge_weight=edge_weight,
            ).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True).sigmoid())

        loss_dgi = 0.
        for pos_h, neg_h, summary in zip(pos_hs, neg_hs, summaries):
            summary = summary.expand_as(pos_h)
            loss_dgi += F.binary_cross_entropy_with_logits(
                input=torch.cat([
                    self.M(pos_h, summary).squeeze(dim=-1),
                    self.M(neg_h, summary).squeeze(dim=-1),
                ]),
                target=torch.cat([
                    torch.ones(pos_h.shape[0]),
                    torch.zeros(neg_h.shape[0]),
                ]).to(summary.device),
            )

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss_cs = (pos_reg_loss - neg_reg_loss)

        loss = loss_dgi + self.alpha * loss_cs

        self.log("loss/dgi", loss_dgi)
        self.log("loss/pos_cs", pos_reg_loss)
        self.log("loss/neg_cs", neg_reg_loss)
        self.log("loss/cs", loss_cs)
        self.log("loss/total", loss)

        return loss
