"""Implementation HDGI model."""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, Sequential

from mssl.models.base import BaseGNN


class SemanticAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(SemanticAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)

    def forward(self, x: torch.Tensor, P: int) -> torch.Tensor:
        h = torch.mm(x, self.W)

        h_prime = (h + self.b.repeat(h.size()[0], 1)).tanh()

        s_attns = torch.mm(h_prime, torch.t(self.q)).view(P, -1)
        N = s_attns.size()[1]
        s_attns = s_attns.mean(dim=1, keepdim=True)
        s_attns = F.softmax(s_attns, dim=0)

        s_attns = s_attns.view(P, 1, 1)
        s_attns = s_attns.repeat(1, N, self.in_features)

        input_embedding = x.view(P, N, self.in_features)

        h_embedding = torch.mul(input_embedding, s_attns)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding


class HGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        attention_dim: int,
        P: int,
    ):
        """Heterogeneous GCN."""
        super(HGCN, self).__init__()

        self.convs = nn.ModuleList([
            Sequential("x, edge_index", [
                (GCNConv(in_dim, out_dim), "x, edge_index -> x"),
                nn.PReLU(),
            ])
            for _ in range(P)
        ])
        self.P = P

        self.sla = SemanticAttentionLayer(out_dim, attention_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: List[torch.Tensor],
    ) -> torch.Tensor:
        h_gcn = []
        for conv, edge_index in zip(self.convs, edge_indices):
            h_gcn.append(conv(x, edge_index))

        x = self.sla(x=torch.cat(h_gcn, dim=0), P=self.P)
        return x


class HDGI(BaseGNN):
    """Implementation based on:
    `https://github.com/YuxiangRen/Heterogeneous-Deep-Graph-Infomax`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_dim: int,
        P: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters()

        self.hgnn = HGCN(
            in_dim=in_channels,
            out_dim=out_channels,
            attention_dim=attention_dim,
            P=P,
        )
        self.M = nn.Bilinear(out_channels, out_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.hgnn.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "HDGI":
        return cls(
            in_channels=data["Node"].x.size(-1),
            out_channels=hparams["emb_dim"],
            attention_dim=hparams["attention_dim"],
            P=len(data.edge_types),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        return self.hgnn(graph["Node"].x, list(graph.edge_index_dict.values()))

    def training_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        x = batch["Node"].x
        edge_indices = list(batch.edge_index_dict.values())

        h_pos = self.hgnn(x, edge_indices)
        h_neg = self.hgnn(
            x=x[torch.randperm(x.size(0), device=x.device)],
            edge_indices=edge_indices,
        )
        c = h_pos.mean(dim=0, keepdim=True).sigmoid().expand_as(h_pos)

        logits = torch.cat([
            self.M(h_pos, c).squeeze(dim=-1),
            self.M(h_neg, c).squeeze(dim=-1),
        ], dim=0)
        labels = torch.cat([
            torch.ones(x.shape[0]),
            torch.zeros(x.shape[0]),
        ], dim=0).to(logits.device)

        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels)

        return loss
