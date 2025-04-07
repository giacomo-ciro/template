from __future__ import annotations

from abc import abstractmethod

import lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import get_class
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW

from template.scheduler import load_scheduler


def load_model(cfg: DictConfig) -> BaseModel:
    return get_class(cfg.model._target_)(cfg)


class BaseModel(pl.LightningModule):
    """Shared training/validation/test logic. Subclasses implement forward and compute_loss."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, batch: dict) -> dict:
        pass

    @abstractmethod
    def compute_loss(self, batch: dict, outdict: dict) -> tuple[torch.Tensor, dict]:
        pass

    def _common_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        outdict = self(batch)
        return self.compute_loss(batch, outdict)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self._common_step(batch)

        log_dict = {"train/" + k: v for k, v in loss_dict.items()}
        log_dict["train/lr"] = self.optimizers().param_groups[0]["lr"]
        self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self._common_step(batch)
        self.log_dict(
            {"val/" + k: v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        _, loss_dict = self._common_step(batch)
        self.log_dict(
            {"test/" + k: v for k, v in loss_dict.items()}, on_step=False, on_epoch=True
        )

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": load_scheduler(self.cfg, optimizer, total_steps),
        }


class MLP(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        model_cfg = cfg.model
        self.ffnn = nn.Sequential(
            nn.Linear(model_cfg.in_dim, model_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_cfg.hidden_dim, model_cfg.out_dim),
        )

    def forward(self, batch: dict) -> dict:
        return {"pred": self.ffnn(batch["x"])}

    def compute_loss(self, batch: dict, outdict: dict) -> tuple[torch.Tensor, dict]:
        loss = F.mse_loss(outdict["pred"], batch["y"])
        return loss, {"loss": loss.detach()}
