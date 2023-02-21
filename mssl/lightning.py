import logging
import warnings
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train_silent_without_validation(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    num_epochs: int,
) -> torch.Tensor:
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", ".*has shuffling enabled.*")

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        check_val_every_n_epoch=None,
        limit_val_batches=0.0,
        enable_model_summary=False,
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        inference_mode=False,
        deterministic="warn",
    )

    trainer.fit(model=model, datamodule=datamodule)

    embeddings = trainer.predict(model=model, datamodule=datamodule)[0]
    return cast(torch.Tensor, embeddings)


def train(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    checkpoint_dir: Path,
    logs_dir: Path,
    hparams: dict,
):
    model_chkpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=str(hparams["seed"]),
        monitor="val/f1_macro",
        verbose=True,
        mode="max",
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(
            save_dir=logs_dir.as_posix(),
            name=None,
            default_hp_metric=False,
        ),
        callbacks=[model_chkpt],
        max_epochs=hparams["max_epochs"],
        log_every_n_steps=1,
        check_val_every_n_epoch=hparams["check_val_every_n_epoch"],
        num_sanity_val_steps=0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        inference_mode=False,
        deterministic="warn",
    )

    trainer.fit(model=model, datamodule=datamodule)

    load_kwargs = {}

    if model.__class__.__name__ == "DeepWalk":
        load_kwargs["edge_index"] = datamodule.data[
            "Node", hparams["edge_type"], "Node"
        ].edge_index
    elif model.__class__.__name__ == "FlattenedGraphDeepWalk":
        load_kwargs["edge_index"] = datamodule.data.to_homogeneous().edge_index

    best_model = model.__class__.load_from_checkpoint(
        checkpoint_path=model_chkpt.best_model_path,
        **load_kwargs,
    )

    embeddings = trainer.predict(model=best_model, datamodule=datamodule)[0]

    return embeddings
