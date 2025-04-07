from __future__ import annotations

import time
from datetime import timedelta

import lightning as pl
from lightning.pytorch.callbacks import RichProgressBar
from rich.progress import ProgressColumn, Task
from rich.text import Text


# Elapsed and remaining time of the whole run, shown next to the step counter (the
# epoch's own elapsed and remaining time are shown after the batch counter). Rendered
# by rich's refresh thread, so the clock keeps ticking between slow batches.
class RunTimeColumn(ProgressColumn):
    max_refresh = 0.5

    def __init__(self, progress_bar: StepProgressBar) -> None:
        super().__init__()
        self.progress_bar = progress_bar

    def render(self, task: Task) -> Text:
        bar = self.progress_bar
        # only on the training bar, and not during the validation run before fit
        if task.id != bar.train_progress_bar_id or bar.train_start_time is None:
            return Text("")
        elapsed = time.time() - bar.train_start_time
        step, max_steps = bar.trainer.global_step, bar.trainer.max_steps
        # average time per optimizer step so far, validation included
        remaining = (
            "-:--:--"
            if step == 0
            else str(timedelta(seconds=int(elapsed / step * (max_steps - step))))
        )
        return Text(
            f"{timedelta(seconds=int(elapsed))} • {remaining}", style=bar.theme.time
        )


class StepProgressBar(RichProgressBar):
    def __init__(self) -> None:
        super().__init__()
        self.train_start_time: float | None = None

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.train_start_time = time.time()
        super().on_train_start(trainer, pl_module)

    def configure_columns(self, trainer: pl.Trainer) -> list:
        columns = super().configure_columns(trainer)
        # right after the description (Step N/max_steps)
        return [columns[0], RunTimeColumn(self), *columns[1:]]

    # Lightning caps the last epoch's bar at the remaining optimizer steps but compares
    # them to batches; with gradient accumulation each step takes several batches.
    @property
    def total_train_batches(self) -> int | float:
        remaining_batches = (
            self.trainer.max_steps - self.trainer.global_step
        ) * self.trainer.accumulate_grad_batches
        return min(self.trainer.num_training_batches, remaining_batches)

    def _get_train_description(self, current_epoch: int) -> str:
        desc = f"Step {self.trainer.global_step}/{self.trainer.max_steps}"
        if len(self.validation_description) > len(desc):
            desc = f"{desc:{len(self.validation_description)}}"
        return desc

    # The bar tracks batches within the current epoch (so gradient accumulation is
    # visible), the description tracks optimizer steps over the whole run. The base
    # class only sets the description at epoch start/end, so refresh it after every batch.
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.progress is not None and self.train_progress_bar_id is not None:
            self.progress.update(
                self.train_progress_bar_id,
                description=self._get_train_description(trainer.current_epoch),
            )
            self.refresh()

    # no metrics in the bar; they are read from the logger
    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict:
        return {}
