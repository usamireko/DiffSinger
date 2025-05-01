import abc
import pathlib
from fnmatch import fnmatch
from typing import Union, Optional, Callable, Any

import lightning.pytorch
import torch
import tqdm
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning_utilities.core.rank_zero import rank_zero_info
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Metric, MeanMetric

from lib.config.schema import ModelConfig, TrainingConfig
from lib.exponential_moving_average import ExponentialMovingAverageV2
from lib.reflection import build_optimizer_from_config, build_lr_scheduler_from_config
from .dataset import BaseDataset, DynamicBatchSampler


class BaseLightningModule(lightning.pytorch.LightningModule, abc.ABC):
    def __init__(
            self,
            binary_data_dir: pathlib.Path,
            model_config: ModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.exponential_moving_average = None
        self.binary_data_dir = binary_data_dir
        self.model_config = model_config
        self.training_config = training_config
        self.model: nn.Module = None
        self.build_model()
        self.losses = nn.ModuleDict()
        self.metrics = nn.ModuleDict()
        self.val_losses: dict[str, Metric] = {  # use built-in dict to not be printed in the model summary
            "total_loss": MeanMetric()
        }
        self.build_losses_and_metrics()
        if len(self.losses) == 0:
            raise ValueError("No losses defined.")
        self.freeze_parameters()  # caution: this can break when resuming training
        self.print_arch()
        self.train_dataset: BaseDataset = None
        self.valid_dataset: BaseDataset = None
        self.train_sampler: DynamicBatchSampler = None
        self.valid_sampler: DynamicBatchSampler = None
        
        self.enabled_ema = training_config.use_ema


    @abc.abstractmethod
    def build_model(self):
        
        self.exponential_moving_average = ExponentialMovingAverageV2(model=self.model, decay=self.training_config.ema_decay,
                         ignored_layers=self.training_config.ema_ignored_layers) if self.enabled_ema else None
        if self.enabled_ema:
            self.EMA.register()
        pass

    @abc.abstractmethod
    def build_losses_and_metrics(self):
        pass

    @abc.abstractmethod
    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]):
        pass

    def print_arch(self):
        rank_zero_info(f"Model: {self.model}")
        rank_zero_info(f"Losses: {self.losses}")
        rank_zero_info(f"Metrics: {self.metrics}")

    def freeze_parameters(self):
        if not self.training_config.finetuning.freezing_enabled:
            return
        frozen_count = 0
        frozen_param_patterns = self.training_config.finetuning.frozen_params
        for name, parameter in self.named_parameters():
            for pattern in frozen_param_patterns:
                if fnmatch(name, pattern):
                    parameter.requires_grad = False
                    frozen_count += 1
                    break
        rank_zero_info(f"Freezing {frozen_count} parameter(s).")

    def load_from_pretrained_model(self, pretrained_model_path: pathlib.Path):
        source_model = torch.load(
            pretrained_model_path, map_location=self.device, weights_only=True
        )
        source_state_dict=source_model["state_dict"]
        if self.enabled_ema:
            if source_model.get("ema_weights", None) is not None:
                load_fn=0
                ema_source_state_dict=source_model["ema_weights"]
            else:
                load_fn=1


        includes = self.training_config.finetuning.pretraining_include_params
        excludes = self.training_config.finetuning.pretraining_exclude_params
        if includes:
            include_names = set()
            for pattern in includes:
                include_names.update(
                    name for name in source_state_dict.keys() if fnmatch(name, pattern)
                )
            source_state_dict = {k: v for k, v in source_state_dict.items() if k in include_names}
            if self.enabled_ema :
                if load_fn==0:
                    ema_source_state_dict = {k: v for k, v in ema_source_state_dict.items() if k in include_names}
        if excludes:
            exclude_names = set()
            for pattern in excludes:
                exclude_names.update(
                    name for name in source_state_dict.keys() if fnmatch(name, pattern)
                )
            source_state_dict = {k: v for k, v in source_state_dict.items() if k not in exclude_names}
            if self.enabled_ema :
                if load_fn==0:
                    ema_source_state_dict = {k: v for k, v in ema_source_state_dict.items() if k not in exclude_names}
        target_state_dict = self.state_dict()
        errors = []
        for name in list(source_state_dict.keys()):
            if name not in target_state_dict:
                del source_state_dict[name]
            source_param = source_state_dict[name]
            target_param = target_state_dict[name]
            if source_param.shape != target_param.shape:
                errors.append((name, tuple(source_param.shape), tuple(target_param.shape)))
                del source_state_dict[name]
        if errors:
            raise RuntimeError(
                f"Pretrained model '{pretrained_model_path}' has {len(errors)} mismatched parameter(s):\n"
                + "\n".join(
                    f"  {name}: source {source_shape}, target {target_shape}"
                    for name, source_shape, target_shape in errors
                )
            )
        self.load_state_dict(source_state_dict, strict=False)
        if self.enabled_ema:
            if load_fn==1:
                self.exponential_moving_average.re_register()
            else:
                ema_s=self.exponential_moving_average.save_state_dict()
                ema_s_key=set(ema_s.keys())
                ema_source_state_dict = {k: v for k, v in ema_source_state_dict.items() if k in ema_s_key}
                ema_s.update(ema_source_state_dict)
                self.exponential_moving_average.load_state_dict(ema_s)
        rank_zero_info(
            f"Loaded {len(source_state_dict)} parameter(s) from '{pretrained_model_path}'"
        )

    def register_loss(self, name: str, loss: nn.Module):
        if name in self.losses:
            raise ValueError(f"Loss '{name}' already registered.")
        self.losses[name] = loss
        self.val_losses[name] = MeanMetric()  # for validation logging

    def register_metric(self, name: str, metric: Metric):
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' already registered.")
        self.metrics[name] = metric

    def build_train_dataset(self):
        return BaseDataset(self.binary_data_dir, "train")

    def build_valid_dataset(self):
        return BaseDataset(self.binary_data_dir, "valid")

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise ValueError("This module only supports the 'fit' stage.")
        self.train_dataset = self.build_train_dataset()
        self.valid_dataset = self.build_valid_dataset()

    def train_dataloader(self):
        dataloader_config = self.training_config.dataloader
        self.train_sampler = DynamicBatchSampler(
            self.train_dataset,
            max_batch_size=dataloader_config.max_batch_size,
            max_batch_frames=dataloader_config.max_batch_frames,
            sort_by_len=True,
            frame_count_grid=dataloader_config.frame_count_grid,
            batch_count_multiple_of=self.training_config.trainer.accumulate_grad_batches,
            reassign_batches=True,
            shuffle_batches=False,
            seed=42,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate,
            batch_sampler=self.train_sampler,
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor if dataloader_config.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=dataloader_config.num_workers > 0
        )

    def val_dataloader(self):
        dataloader_config = self.training_config.dataloader
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            collate_fn=self.valid_dataset.collate,
            batch_sampler=DynamicBatchSampler(
                self.valid_dataset,
                max_batch_size=dataloader_config.max_val_batch_size,
                max_batch_frames=dataloader_config.max_val_batch_frames,
                sort_by_len=False,
                reassign_batches=False,
                shuffle_batches=False,
            ),
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor if dataloader_config.num_workers > 0 else None,
            persistent_workers=dataloader_config.num_workers > 0
        )

    def on_train_epoch_start(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

    def training_step(self, sample: dict[str, torch.Tensor], batch_index: int):
        losses = self.forward_model(sample, infer=False)
        total_loss = sum(losses.values())
        log_outputs = {
            **losses,
            "batch_size": sample["size"],
        }
        # logs to progress bar
        self.log_dict(log_outputs, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # logs to tensorboard
        if self.global_step % self.training_config.trainer.log_every_n_steps == 0:
            tb_log = {f"training/{k}": v for k, v in log_outputs.items()}
            tb_log["training/lr"] = self.lr_schedulers().get_last_lr()[0]
            self.logger.log_metrics(tb_log, step=self.global_step)
        return total_loss

    def on_validation_epoch_start(self):
        if self.enabled_ema:
            self.exponential_moving_average.apply_shadow()
        for metric in self.val_losses.values():
            # self.val_losses is a built-in dict, so we need to move them to device manually
            metric.to(self.device)
            metric.reset()

    def validation_step(self, sample: dict[str, torch.Tensor], batch_index: int):
        if sample["size"] == 0:
            return
        save_obj = {}
        with torch.autocast(self.device.type, enabled=False):
            losses = self.forward_model(sample, infer=False)
            if min(sample["indices"]) < self.training_config.validation.max_plots:
                outputs = self.forward_model(sample, infer=True)
                save_obj["sample"] = sample
                save_obj["outputs"] = outputs
                filename = f"validation_step{self.global_step}_rank{self.global_rank}_batch{batch_index}.pt"
                torch.save(
                    obj=save_obj,
                    f=pathlib.Path(self.logger.log_dir) / filename,
                )
            losses = {
                "total_loss": sum(losses.values()),
                **losses,
            }
            for k, v in losses.items():
                self.val_losses[k].update(v, weight=sample["size"])

    def on_validation_epoch_end(self):
        if self.enabled_ema:
            self.exponential_moving_average.restore()
        loss_vals = {k: v.compute() for k, v in self.val_losses.items()}
        metric_vals = {k: v.compute() for k, v in self.metrics.items()}
        self.log_dict(
            {**loss_vals, **metric_vals},
            on_epoch=True, prog_bar=False, logger=False, sync_dist=True
        )
        if self.global_rank != 0:
            return
        self.logger.log_metrics({f"validation/{k}": v for k, v in loss_vals.items()}, step=self.global_step)
        self.logger.log_metrics({f"metrics/{k}": v for k, v in metric_vals.items()}, step=self.global_step)
        filelist = list(pathlib.Path(self.logger.log_dir).glob(f"validation_step{self.global_step}_rank*_batch*.pt"))
        with torch.autocast(self.device.type, enabled=False):
            for file in tqdm.tqdm(filelist, desc="Plotting", leave=False):
                obj = torch.load(file, map_location=self.device, weights_only=True)
                sample = obj["sample"]
                outputs = obj["outputs"]
                self.plot_validation_results(sample, outputs)
                file.unlink()

    def configure_optimizers(self):
        optimizer = build_optimizer_from_config(self.model, self.training_config.optimizer)
        scheduler = build_lr_scheduler_from_config(optimizer, self.training_config.lr_scheduler)
        interval = self.training_config.lr_scheduler.unit
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if interval != self.training_config.trainer.unit:
                # ReduceLROnPlateau requires the scheduler to synchronize with the validation period
                raise ValueError(
                    f"ReduceLROnPlateau scheduler requires training.lr_scheduler.unit and training.trainer.unit "
                    f"to be the same, got '{interval}' and '{self.training_config.trainer.unit}'."
                )
            # Call scheduler.step() after each validation
            frequency = self.training_config.trainer.val_every_n_units
            monitor = self.training_config.lr_scheduler.monitor
            if monitor not in self.val_losses and monitor not in self.metrics:
                raise ValueError(
                    f"Invalid monitor '{monitor}' for ReduceLROnPlateau scheduler. Should be one of "
                    f"losses {list(self.val_losses.keys())} or metrics {list(self.metrics.keys())}."
                )
        else:
            frequency = 1
            monitor = None
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": frequency,
                "monitor": monitor,
                "strict": False,  # in case the candidates are empty after resuming
            }
        }


    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().optimizer_step(epoch=epoch, batch_idx=batch_idx, optimizer=optimizer, optimizer_closure=optimizer_closure)
        if self.enabled_ema:
            self.exponential_moving_average.update()

    def on_save_checkpoint(self, checkpoint):
        if self.enabled_ema:
            checkpoint['ema_weights'] = self.exponential_moving_average.save_state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.enabled_ema:
            self.exponential_moving_average.load_state_dict(checkpoint.pop('ema_weights'))
