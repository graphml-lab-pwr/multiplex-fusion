"""Script for applying and evaluating trainable fusion approaches."""
import json
from pathlib import Path
from typing import Callable

import click
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm

from mssl.fusion import AttentionFusion, ConcatLinFusion, LookupFusion
from mssl.fusion.base import Fusion
from mssl.lightning import train
from mssl.tasks import evaluate_all_node_tasks, evaluate_node_classification
from mssl.utils import barlow_twins_loss, parse_seeds


class FusionDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_file: Path,
        embeddings_dir: Path,
        seed: int,
    ):
        super().__init__()
        self.data = torch.load(dataset_file)
        self.layer_embs = {
            et: torch.load(embeddings_dir / f"{et}_{seed}.pt")
            for _, et, _ in self.data.edge_types
        }

    def _dataloader(self):
        return DataLoader(
            dataset=[(self.layer_embs, self.data)],
            collate_fn=lambda x: x[0],
            batch_size=1,
        )

    def train_dataloader(self):
        return self._dataloader()

    def val_dataloader(self):
        return self._dataloader()

    def predict_dataloader(self):
        return self._dataloader()


class FusionWrapper(pl.LightningModule):
    """Implementation for trainable fusion approaches wrapper."""

    def __init__(
        self,
        fusion: Fusion,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.fusion = fusion
        self.loss_fn = loss_fn

        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], HeteroData],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = 0

        zs, _ = batch

        z = self.fusion(zs)

        for z_i in zs.values():
            loss += self.loss_fn(z, z_i)

        loss /= len(zs.keys())

        return loss

    def training_epoch_end(self, outputs):
        """Summarizes training metrics."""
        avg_loss = sum(o["loss"] for o in outputs) / len(outputs)

        self.log("step", float(self.trainer.current_epoch))
        self.log("loss/avg", avg_loss)

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], HeteroData],
        batch_idx: int,
    ) -> tuple[torch.Tensor, HeteroData]:
        zs, graph = batch
        z = self.fusion(zs).cpu()
        return z.cpu(), graph.cpu()

    def validation_epoch_end(self, outputs):
        """Evaluates embedding in given task."""
        assert len(outputs) == 1
        z, graph = outputs[0]

        mtr = evaluate_node_classification(
            z=z,
            y=graph["Node"].y,
            train_mask=graph["Node"].train_mask,
            val_mask=graph["Node"].val_mask,
            test_mask=graph["Node"].test_mask,
            seed=self.seed,
        )

        self.log("step", float(self.trainer.current_epoch))

        for split_name, metrics in mtr.items():
            self.log(f"{split_name}/acc", metrics["accuracy"])
            self.log(f"{split_name}/f1_macro", metrics["macro avg"]["f1-score"])
            self.log(f"{split_name}/f1_micro", metrics["micro avg"]["f1-score"])
            self.log(f"{split_name}/auc", metrics["auc"])

        # Log 2D PCA projections of embeddings
        fig, ax = plt.subplots(figsize=(10, 10))

        z_scale = StandardScaler().fit_transform(z)
        z2d = PCA(n_components=2).fit_transform(z_scale)
        sns.scatterplot(x=z2d[:, 0], y=z2d[:, 1], hue=graph["Node"].y)

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
        batch: tuple[dict[str, torch.Tensor], HeteroData],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        zs, _ = batch
        return self.fusion(zs).cpu()

    def configure_optimizers(self):
        """Prepares the optimizer module."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


def create_fusion_model(
    name: str,
    data: HeteroData,
) -> tuple[Fusion, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    fusion_name, loss_name = name.split("_")

    edge_types = [et for _, et, _ in data.edge_types]

    if loss_name == "BT":
        loss_fn = barlow_twins_loss
    elif loss_name == "MSE":
        loss_fn = F.mse_loss
    else:
        raise ValueError(f"Unknown loss function: '{loss_name}'")

    if fusion_name == "Attention":
        fusion = AttentionFusion(emb_dim=64, edge_types=edge_types)
    elif fusion_name == "ConcatLin":
        fusion = ConcatLinFusion(emb_dim=64, edge_types=edge_types)
    elif fusion_name == "Lookup":
        fusion = LookupFusion(num_nodes=data["Node"].num_nodes, emb_dim=64)
    else:
        raise ValueError(f"Unknown fusion model name: '{fusion_name}'")

    return fusion, loss_fn


@click.command()
@click.option(
    "--seeds",
    help="List of seeds for PRNGs",
    type=str,
    callback=parse_seeds,
)
@click.option(
    "--dataset_file",
    help="Path to the graph dataset",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--embeddings_dir",
    help="Path to directory with embeddings",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--output_dir",
    help="Path to output directory",
    type=click.Path(path_type=Path),
)
def main(
    seeds: list[int],
    dataset_file: Path,
    embeddings_dir: Path,
    output_dir: Path,
):
    """Trains model and saves embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(iterable=seeds, desc="Seeds"):
        data = torch.load(dataset_file)

        for model_name in tqdm(
            iterable=(
                "Attention_BT",
                "Attention_MSE",
                "ConcatLin_BT",
                "ConcatLin_MSE",
                "Lookup_BT",
                "Lookup_MSE",
            ),
            desc="Trainable fusion models",
        ):
            (output_dir / model_name).mkdir(parents=True, exist_ok=True)

            seed_everything(seed, workers=True)

            datamodule = FusionDataModule(
                dataset_file=dataset_file,
                embeddings_dir=embeddings_dir,
                seed=seed,
            )

            fusion_model, loss_fn = create_fusion_model(
                name=model_name,
                data=data,
            )
            model = FusionWrapper(
                fusion=fusion_model,
                loss_fn=loss_fn,
                lr=1e-3,
                weight_decay=5e-4,
                seed=seed,
            )

            z_fused = train(
                model=model,
                datamodule=datamodule,
                checkpoint_dir=output_dir / model_name / "models",
                logs_dir=output_dir / model_name / "logs" / str(seed),
                hparams={
                    "max_epochs": 2_000,
                    "check_val_every_n_epoch": 50,
                    "seed": seed,
                },
            )

            embeddings_out_dir = output_dir / model_name / "embeddings"
            embeddings_out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(obj=z_fused, f=embeddings_out_dir / f"{seed}.pt")

            metrics = evaluate_all_node_tasks(z=z_fused, data=data, seed=seed)

            metrics_dir = output_dir / model_name / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with (metrics_dir / f"{seed}.json").open("w") as fout:
                json.dump(obj=metrics, fp=fout, indent=4)


if __name__ == "__main__":
    main()
