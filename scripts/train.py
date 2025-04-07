from __future__ import annotations

import os
from pathlib import Path

os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import lightning as pl
from omegaconf import DictConfig

from template.datamodule import load_datamodule
from template.model import load_model
from template.trainer import load_trainer


def train(cfg: DictConfig) -> None:
    trainer = load_trainer(cfg)
    datamodule = load_datamodule(cfg)
    model = load_model(cfg)

    # Validate once first for a starting point, then train and test.
    trainer.validate(model, datamodule)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


@hydra.main(
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    train(cfg)


if __name__ == "__main__":
    main()
