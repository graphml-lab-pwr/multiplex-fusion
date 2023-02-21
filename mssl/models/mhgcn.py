"""Implementation MHGCN model."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

from mssl.models.base import BaseGNN


class MHGCN(BaseGNN):
    """Implementation based on: `https://github.com/NSSSJSS/MHGCN`
    """

    def __init__(
        self,
        feature_dim: int,
        emb_dim: int,
        num_relations: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, seed=seed)
        self.save_hyperparameters()

        self.convs = nn.ModuleList([
            GCNConv(feature_dim, emb_dim, normalize=False),
            GCNConv(emb_dim, emb_dim, normalize=False),
        ])

        self.beta = nn.Parameter(torch.FloatTensor(num_relations, 1))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        nn.init.uniform_(self.beta, a=0, b=0.1)

    @classmethod
    def from_hparams(cls, data: HeteroData, hparams: dict) -> "MHGCN":
        return cls(
            feature_dim=data["Node"].x.size(-1),
            emb_dim=hparams["emb_dim"],
            num_relations=len(data.edge_types),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            seed=hparams["seed"],
        )

    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        full_adj = 0
        num_nodes = graph["Node"].num_nodes

        for idx, edge_type in enumerate(graph.edge_types):
            ei = graph[edge_type].edge_index
            spt = torch.sparse_coo_tensor(
                indices=ei,
                values=self.beta[idx].repeat(ei.shape[1]),
                size=(num_nodes, num_nodes),
            )

            if idx == 0:
                full_adj = spt
            else:
                full_adj = full_adj + spt

        full_adj = full_adj.coalesce()

        edge_index = full_adj.indices()
        edge_weight = full_adj.values()

        h = self.convs[0](
            x=graph["Node"].x,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        z = self.convs[1](
            x=h,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        return (h + z) / 2

    def training_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single training step."""
        pos_ei = []
        neg_ei = []

        for edge_type in batch.edge_types:
            pos_ei.append(batch[edge_type].edge_index)
            neg_ei.append(
                negative_sampling(
                    edge_index=pos_ei[-1],
                    num_nodes=batch["Node"].num_nodes,
                    force_undirected=True,
                )
            )

        pos_ei = torch.cat(pos_ei, dim=-1)
        neg_ei = torch.cat(neg_ei, dim=-1)

        z = self(batch)

        pos_score = (z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=-1)
        neg_score = -(z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=-1)

        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(neg_score).mean()

        return loss
