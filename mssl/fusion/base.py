"""Base class implementation for fusion models."""
from abc import abstractmethod

import torch
from torch import nn


class Fusion(nn.Module):

    def __init__(self, edge_types: list[str]):
        super().__init__()
        self.edge_types = edge_types

    @abstractmethod
    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def _to_ordered(self, zs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [zs[et] for et in self.edge_types]
