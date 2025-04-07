from __future__ import annotations

from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from template.progress_bar import StepProgressBar


def load_trainer(cfg: DictConfig) -> Trainer:
    return Trainer(cfg)


class Trainer(pl.Trainer):
    def __init__(self, cfg: DictConfig):
        # to use tensor cores on recent GPUs
        torch.set_float32_matmul_precision("high")

        self.cfg = cfg
        trainer_cfg = cfg.trainer
        super().__init__(
            max_epochs=-1,
            max_steps=trainer_cfg.max_steps,
            val_check_interval=trainer_cfg.val_every_n_steps
            * trainer_cfg.accumulate_grad_batches,
            check_val_every_n_epoch=None,
            log_every_n_steps=trainer_cfg.log_every_n_steps,
            accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
            gradient_clip_val=trainer_cfg.gradient_clip_val,
            enable_checkpointing=trainer_cfg.enable_checkpointing,
            logger=self._init_logger(),
            callbacks=self._init_callbacks(),
            num_sanity_val_steps=0,
        )

    def _init_callbacks(self) -> list[pl.Callback]:
        callbacks = [StepProgressBar()]
        if self.cfg.trainer.enable_checkpointing:
            callbacks.append(self._init_checkpointer())
        return callbacks

    def _init_checkpointer(self):
        # overwrites a single best.ckpt whenever val/loss improves
        return ModelCheckpoint(
            dirpath=Path(self.cfg.paths.checkpoints) / self.cfg.run_name,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            enable_version_counter=False,
        )

    def _init_logger(self) -> Logger | None:
        logger_cfg = self.cfg.get("logger")
        if logger_cfg is None:
            return None
        if logger_cfg.name == "wandb":
            logger = WandbLogger(
                project=logger_cfg.project,
                entity=logger_cfg.entity,
                name=self.cfg.run_name,
            )
        elif logger_cfg.name == "aim":
            from aim.pytorch_lightning import AimLogger

            logger = AimLogger(
                repo=logger_cfg.repo,
                experiment=self.cfg.run_name,
                log_system_params=False,
            )
        else:
            raise ValueError()
        logger.log_hyperparams(OmegaConf.to_container(self.cfg, resolve=True))
        return logger
