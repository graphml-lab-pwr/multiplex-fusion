"""Implementations of naive fusion mechanisms."""
import torch
from torch import nn

from mssl.fusion.base import Fusion


class MinFusion(Fusion):

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(self._to_ordered(zs), dim=-1).amin(dim=-1)


class MeanFusion(Fusion):

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(self._to_ordered(zs), dim=-1).mean(dim=-1)


class MaxFusion(Fusion):

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(self._to_ordered(zs), dim=-1).amax(dim=-1)


class SumFusion(Fusion):

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(self._to_ordered(zs), dim=-1).sum(dim=-1)


class ConcatFusion(Fusion):

    def forward(self, zs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(self._to_ordered(zs), dim=-1)
