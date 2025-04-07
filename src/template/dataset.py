from __future__ import annotations

import torch
from hydra.utils import get_class
from omegaconf import DictConfig
from torch.utils.data import Dataset


def load_dataset(cfg: DictConfig, split: str) -> Dataset:
    return get_class(cfg.dataset._target_)(cfg, split)


# Replace with a class that reads e.g. {cfg.paths.data}/processed/{split}/.
class RandomDataset(Dataset):
    SPLITS = ("train", "val", "test")

    def __init__(self, cfg: DictConfig, split: str):
        ds_cfg = cfg.dataset
        w = torch.randn(
            ds_cfg.in_dim, ds_cfg.out_dim, generator=torch.Generator().manual_seed(0)
        )
        g = torch.Generator().manual_seed(self.SPLITS.index(split))
        self.x = torch.randn(ds_cfg.num_samples, ds_cfg.in_dim, generator=g)
        self.y = self.x @ w

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        return {"x": self.x[idx], "y": self.y[idx]}
