import base64
from collections import deque
from typing import Literal, Any

import lightning.pytorch
import lightning.pytorch.callbacks
import sympy
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import Tensor

from training.pl_module_base import BaseLightningModule


class PeriodicModelCheckpoint(lightning.pytorch.callbacks.ModelCheckpoint):
    def __init__(
            self,
            dirpath: str,
            prefix: str,
            unit: Literal["step", "epoch"],
            every_n_units: int,
            since_m_units: int = 0,
            save_last_k: int = 1,
            save_weights_only: bool = False,
    ):
        if unit == "step":
            filename = f"{prefix}-steps={{step:07d}}"
            every_n_train_steps = every_n_units
            every_n_epochs = None
            save_on_epoch_end = False
        elif unit == "epoch":
            filename = f"{prefix}-epochs={{epoch:04d}}"
            every_n_train_steps = None
            every_n_epochs = every_n_units
            save_on_epoch_end = True
        else:
            raise ValueError(f"Unit must be 'step' or 'epoch', got '{unit}'.")
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            verbose=True,
            save_last=False,
            save_top_k=-1,
            save_weights_only=save_weights_only,
            auto_insert_metric_name=False,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_epoch_end,
            enable_version_counter=False,
        )
        self.unit = unit
        self.since_m_units = since_m_units
        self.save_last_k = save_last_k
        self.last_k_models = deque()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["last_k_models"] = list(self.last_k_models)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_k_models = deque(state.pop("last_k_models", []))
        super().load_state_dict(state)

    def on_train_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs
    ) -> None:
        if self.unit == "step" and trainer.global_step < self.since_m_units:
            return
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

    def on_train_epoch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule
    ) -> None:
        if self.unit == "epoch" and trainer.current_epoch + 1 < self.since_m_units:
            return
        super().on_train_epoch_end(trainer, pl_module)

    def format_checkpoint_name(self, metrics: dict[str, Tensor], *args, **kwargs) -> str:
        if self.unit == "epoch":
            metrics = metrics.copy()
            metrics["epoch"] += 1
        return super().format_checkpoint_name(metrics, *args, **kwargs)

    def _save_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            filepath: str
    ) -> None:
        @rank_zero_only
        def progress_bar_print(s):
            trainer.progress_bar_callback.print(s)

        if self.save_last_k == 0:
            return
        self.last_k_models.append(filepath)
        super()._save_checkpoint(trainer, filepath)
        progress_bar_print("Saved checkpoint: " + filepath)
        if self.save_last_k == -1:
            return
        while len(self.last_k_models) > self.save_last_k:
            self._remove_checkpoint(trainer, self.last_k_models.popleft())

    def _remove_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            filepath: str
    ) -> None:
        @rank_zero_only
        def progress_bar_print(s):
            trainer.progress_bar_callback.print(s)

        super()._remove_checkpoint(trainer, filepath)
        progress_bar_print("Removed checkpoint: " + filepath)


class ExpressionModelCheckpoint(lightning.pytorch.callbacks.ModelCheckpoint):
    def __init__(
            self,
            dirpath: str,
            prefix: str,
            expression: str,
            mode: Literal["min", "max"],
            save_top_k: int = 1,
            save_weights_only: bool = False,
    ):
        parsed_expr = sympy.sympify(expression)
        if isinstance(parsed_expr, sympy.Number):
            raise ValueError(f"Expression '{expression}' is a constant.")
        if isinstance(parsed_expr, sympy.Symbol):
            metric_name = expression
        else:
            metric_name = "expr"
        metric_key = f"expr_{base64.b16encode(expression.encode('utf8')).decode()}"
        super().__init__(
            dirpath=dirpath,
            filename=f"{prefix}-{metric_name}={{{metric_key}:06.3f}}",
            monitor=metric_key,
            verbose=False,
            save_last=False,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        self.expression = parsed_expr
        self.metric_key = metric_key

    def _save_topk_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            monitor_candidates: dict[str, Tensor]
    ) -> None:
        pl_module: BaseLightningModule = trainer.lightning_module
        candidates = {
            **{k: v.compute() for k, v in pl_module.validation_losses.items()},
            **{k: v.compute() for k, v in pl_module.validation_metrics.items()},
        }
        eval_result = self.expression.evalf(subs=candidates)
        if not isinstance(eval_result, sympy.Number):
            raise ValueError(
                f"Expression '{self.expression}' not fully evaluated. "
                f"Valid candidates: {list(candidates.keys())}"
            )
        eval_result = torch.tensor(float(eval_result), dtype=torch.float32)
        monitor_candidates[self.metric_key] = eval_result
        super()._save_topk_checkpoint(trainer, monitor_candidates)

    def _save_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            filepath: str
    ) -> None:
        @rank_zero_only
        def progress_bar_print(s):
            trainer.progress_bar_callback.print(s)

        super()._save_checkpoint(trainer, filepath)
        progress_bar_print("Saved checkpoint: " + filepath)

    def _remove_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            filepath: str
    ) -> None:
        @rank_zero_only
        def progress_bar_print(s):
            trainer.progress_bar_callback.print(s)

        super()._remove_checkpoint(trainer, filepath)
        progress_bar_print("Removed checkpoint: " + filepath)
