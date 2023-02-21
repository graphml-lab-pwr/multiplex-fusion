"""Base class implementation for GNN."""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

from mssl.tasks import evaluate_node_classification


class BaseGNN(pl.LightningModule, ABC):
    """Implementation of base class for all checked models."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    def forward(self, graph: HeteroData) -> torch.Tensor:
        return self.forward_repr(graph)

    @abstractmethod
    def forward_repr(self, graph: HeteroData) -> torch.Tensor:
        pass

    def training_epoch_end(self, outputs):
        """Summarizes training metrics."""
        avg_loss = sum(o["loss"] for o in outputs) / len(outputs)

        self.log("step", float(self.trainer.current_epoch))
        self.log("loss/avg", avg_loss)

    def validation_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ):
        """Extracts node embeddings for downstream model."""
        z = self.forward_repr(batch).cpu()
        data = batch.cpu()

        return z, data

    def validation_epoch_end(self, outputs):
        """Evaluates embedding in given task."""
        assert len(outputs) == 1
        z, data = outputs[0]

        self.log("step", float(self.trainer.current_epoch))

        train_mask = data["Node"].train_mask.cpu()
        val_mask = data["Node"].val_mask.cpu()
        test_mask = data["Node"].test_mask.cpu()

        y = data["Node"].y.cpu()

        mtr = evaluate_node_classification(
            z=z,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            seed=self.seed,
        )

        for split, metrics in mtr.items():
            self.log(f"{split}/acc", metrics["accuracy"])
            self.log(f"{split}/f1_macro", metrics["macro avg"]["f1-score"])
            self.log(f"{split}/f1_micro", metrics["micro avg"]["f1-score"])
            self.log(f"{split}/auc", metrics["auc"])

        # Log 2D PCA projections of embeddings
        fig, ax = plt.subplots(figsize=(10, 10))

        z_scale = StandardScaler().fit_transform(z)
        z2d = PCA(n_components=2).fit_transform(z_scale)
        sns.scatterplot(x=z2d[:, 0], y=z2d[:, 1], hue=y)

        self.logger.experiment.add_figure(
            tag="z",
            figure=fig,
            global_step=self.trainer.current_epoch,
        )

        # Check embedding params
        self.log("z/min", z.min())
        self.log("z/max", z.max())
        self.log("z/mean", z.mean())
        self.log("z/std", z.std())
        self.log("z/norm", z.norm())

    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        return self.forward_repr(batch).cpu()

    def configure_optimizers(self):
        """Prepares the optimizer module."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
