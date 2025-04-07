from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, OneCycleLR


# Builds the Lightning lr_scheduler config from cfg.scheduler (configs/scheduler/{name}.yaml).
# Every key but `name` is passed to the scheduler; it steps once per optimizer step.
def load_scheduler(
    cfg: DictConfig, optimizer: Optimizer, total_steps: int
) -> dict[str, Any]:
    kwargs = {k: v for k, v in cfg.scheduler.items() if k != "name"}
    name = cfg.scheduler.name

    if name == "ConstantLR":
        scheduler = ConstantLR(optimizer, total_iters=total_steps, **kwargs)
    elif name == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer, max_lr=cfg.optimizer.lr, total_steps=total_steps, **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    return {"scheduler": scheduler, "interval": "step"}
