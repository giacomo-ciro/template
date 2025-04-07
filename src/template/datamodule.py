from __future__ import annotations

import os

import lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset

from template.dataset import load_dataset


def load_datamodule(cfg: DictConfig) -> DataModule:
    return DataModule(cfg)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None) -> None:
        self.train_ds = self._load_split("train")
        self.val_ds = self._load_split("val")
        self.test_ds = self._load_split("test")

    # Splits are fixed on disk. The cap keeps only the first N samples for faster runs.
    def _load_split(self, split: str) -> Dataset:
        ds = load_dataset(self.cfg, split)
        cap = self.cfg.datamodule.split[split]
        return ds if cap < 0 else Subset(ds, range(min(cap, len(ds))))

    def _num_workers(self) -> int:
        # 0 loads in the main process (for debug), N > 0 uses N workers per DataLoader
        num_workers = self.cfg.datamodule.num_workers
        if num_workers >= 0:
            return num_workers
        # -1 uses all CPUs. sched_getaffinity respects SLURM/cgroup allocations (cpu_count reports the whole
        # node) but is Linux-only; divide by num_devices since DDP spawns one DataLoader
        # (and its own worker pool) per GPU process.
        cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        return max(cpus // self.trainer.num_devices, 1)

    def _dataloader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        num_workers = self._num_workers()
        return DataLoader(
            ds,
            batch_size=self.cfg.datamodule.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_ds, shuffle=False)
