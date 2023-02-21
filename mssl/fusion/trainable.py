import torch
from torch import nn

from mssl.fusion.base import Fusion


class AttentionFusion(Fusion):

    def __init__(self, emb_dim: int, edge_types: list[str]):
        super().__init__(edge_types=edge_types)

        self.V = nn.Bilinear(emb_dim, emb_dim, 1, bias=False)
        self.q = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.V.reset_parameters()
        self.q[0].reset_parameters()

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        b = torch.cat(
            [
                self.V(self.q(zs[edge_type]), zs[edge_type]).tanh()
                for edge_type in self.edge_types
            ],
            dim=-1,
        )

        alpha = b.softmax(dim=-1)
        embeddings = torch.stack(self._to_ordered(zs), dim=-1)

        z = (alpha.unsqueeze(dim=1) * embeddings).sum(dim=-1)

        return z


class ConcatLinFusion(Fusion):

    def __init__(self, emb_dim: int, edge_types: list[str]):
        super().__init__(edge_types=edge_types)

        self.lin = nn.Linear(
            in_features=len(edge_types) * emb_dim,
            out_features=emb_dim,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.lin(torch.cat(self._to_ordered(zs), dim=-1))


class LookupFusion(Fusion):

    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__(edge_types=[])  # Not used in this fusion approach

        self.embedding = nn.Embedding(
            num_embeddings=num_nodes,
            embedding_dim=emb_dim,
        )

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.embedding.weight
