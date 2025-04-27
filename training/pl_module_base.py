import abc
import pathlib

import lightning.pytorch
import torch
import tqdm
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric, MeanMetric

from lib.config.schema import ModelConfig, TrainingConfig
from utils import build_object_from_class_name
from .dataset import BaseDataset, DynamicBatchSampler


class BaseLightningModule(lightning.pytorch.LightningModule, abc.ABC):
    def __init__(
            self,
            binary_data_dir: pathlib.Path,
            model_config: ModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.binary_data_dir = binary_data_dir
        self.model_config = model_config
        self.training_config = training_config
        self.model: nn.Module = None
        self.losses = nn.ModuleDict()
        self.validation_losses: dict[str, Metric] = {
            "total_loss": MeanMetric()
        }
        self.validation_metrics: dict[str, Metric] = {}
        self.build_model()
        self.build_losses_and_metrics()
        # TODO: from pretrained model and freezing parameters
        if len(self.losses) == 0:
            raise ValueError("No losses defined.")
        self.print_arch()
        self.train_dataset: BaseDataset = None
        self.valid_dataset: BaseDataset = None
        self.train_sampler: DynamicBatchSampler = None
        self.valid_sampler: DynamicBatchSampler = None

    @abc.abstractmethod
    def build_model(self):
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

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise ValueError("This data module only supports the 'fit' stage.")
        self.train_dataset = BaseDataset(self.binary_data_dir, "train")
        self.valid_dataset = BaseDataset(self.binary_data_dir, "valid")

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
        for metric in self.validation_metrics.values():
            metric.to(self.device)
            metric.reset()
        for metric in self.validation_losses.values():
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
            losses = {
                "total_loss": sum(losses.values()),
                **losses,
            }
            save_obj["losses"] = losses
            save_obj["weight"] = sample["size"]
        filename = f"validation_{self.global_step}_{batch_index}.pt"
        torch.save(
            obj=save_obj,
            f=pathlib.Path(self.logger.log_dir) / filename
        )

    def on_validation_epoch_end(self):
        filelist = list(pathlib.Path(self.logger.log_dir).glob(f"validation_{self.global_step}_*.pt"))
        with torch.autocast(self.device.type, enabled=False):
            for file in tqdm.tqdm(filelist, desc="Plotting", leave=False):
                obj = torch.load(file, map_location=self.device)
                losses = obj["losses"]
                weight = obj["weight"]
                for k, v in losses.items():
                    self.validation_losses[k].update(v, weight=weight)
                if "outputs" in obj:
                    sample = obj["sample"]
                    outputs = obj["outputs"]
                    self.plot_validation_results(sample, outputs)
                file.unlink()
            loss_vals = {k: v.compute() for k, v in self.validation_losses.items()}
            metric_vals = {k: v.compute() for k, v in self.validation_metrics.items()}
        self.log("val_loss", loss_vals["total_loss"], on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.logger.log_metrics({f"validation/{k}": v for k, v in loss_vals.items()}, step=self.global_step)
        self.logger.log_metrics({f"metrics/{k}": v for k, v in metric_vals.items()}, step=self.global_step)

    def configure_optimizers(self):
        # TODO: potential bug if kwargs is not filtered
        optimizer = build_object_from_class_name(
            self.training_config.optimizer.cls,
            torch.optim.Optimizer,
            self.model.parameters(),
            **self.training_config.optimizer.kwargs
        )
        scheduler = build_object_from_class_name(
            self.training_config.lr_scheduler.cls,
            LRScheduler,
            optimizer=optimizer,
            **self.training_config.lr_scheduler.kwargs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.training_config.lr_scheduler.unit,
                "frequency": 1
            }
        }

    @rank_zero_only
    def print_arch(self):
        print(f"Model: {self.model}")
        print(f"Losses: {self.losses}")
