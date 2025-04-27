import os
import pathlib
import shutil
import sys


root_dir = pathlib.Path(__file__).parent.parent.resolve()
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))

import click
import yaml
from pydantic import BaseModel

from lib.config.formatter import ModelFormatter
from lib.config.io import load_raw_config
from lib.config.schema import (
    ConfigurationScope, RootConfig,
    PeriodicCheckpointConfig, ExpressionCheckpointConfig,
)


def _load_and_log_config(config_path: pathlib.Path, scope: int, overrides: list[str] = None) -> RootConfig:
    config = load_raw_config(config_path, overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    formatter = ModelFormatter()
    print(formatter.format(config.model))
    print(formatter.format(config.training))
    return config


def train_model(
        config: RootConfig, pl_module_cls,
        ckpt_save_dir: pathlib.Path, resume_from: pathlib.Path = None
):
    import lightning.pytorch
    import lightning.pytorch.loggers
    from lightning_utilities.core.rank_zero import rank_zero_only
    from training.pl_module_base import BaseLightningModule

    from training.checkpoint import PeriodicModelCheckpoint, ExpressionModelCheckpoint
    from utils.training_utils import DsTQDMProgressBar, get_strategy

    if not issubclass(pl_module_cls, BaseLightningModule):
        raise ValueError("pl_module_cls must be a subclass of BaseLightningModule")

    @rank_zero_only
    def _payload_copy(from_dir: pathlib.Path, to_dir: pathlib.Path):
        shutil.copy(from_dir / "spk_map.json", to_dir)
        shutil.copy(from_dir / "lang_map.json", to_dir)
        shutil.copy(from_dir / "ph_map.json", to_dir)

    @rank_zero_only
    def _config_dump(cfg: BaseModel, dump_path: pathlib.Path):
        with open(dump_path, "w", encoding="utf8") as f:
            yaml.safe_dump(cfg.model_dump(include={"model", "inference"}), f, allow_unicode=True, sort_keys=False)

    binary_data_dir = config.binarizer.binary_data_dir_resolved
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    _payload_copy(binary_data_dir, ckpt_save_dir)
    _config_dump(config, ckpt_save_dir / "config.yaml")
    model_config = config.model
    training_config = config.training

    pl_module = pl_module_cls(binary_data_dir, model_config, training_config)

    if training_config.trainer.unit == "step":
        val_check_interval = (
                training_config.trainer.val_every_n_units * training_config.trainer.accumulate_grad_batches
        )
        check_val_every_n_epoch = None
    elif training_config.trainer.unit == "epoch":
        val_check_interval = None
        check_val_every_n_epoch = training_config.trainer.val_every_n_units
    else:
        raise ValueError(f"Unit must be 'step' or 'epoch', got '{training_config.trainer.unit}'.")
    callbacks = [
        DsTQDMProgressBar()
    ]
    for config in training_config.trainer.checkpoints:
        if config.type == "periodic":
            config: PeriodicCheckpointConfig
            checkpoint = PeriodicModelCheckpoint(
                dirpath=ckpt_save_dir,
                prefix=config.prefix,
                unit=config.unit,
                every_n_units=config.every_n_units,
                since_m_units=config.since_m_units,
                save_last_k=config.save_last_k,
                save_weights_only=config.weights_only,
            )
        elif config.type == "expression":
            config: ExpressionCheckpointConfig
            checkpoint = ExpressionModelCheckpoint(
                dirpath=ckpt_save_dir,
                prefix=config.prefix,
                expression=config.expression,
                mode=config.mode,
                save_top_k=config.save_top_k,
                save_weights_only=config.weights_only,
            )
        else:
            raise ValueError(f"Invalid checkpoint monitor type: {config.type}")
        callbacks.append(checkpoint)
    trainer = lightning.pytorch.Trainer(
        accelerator=training_config.trainer.accelerator,
        # TODO: strategy
        strategy=get_strategy(
            devices=training_config.trainer.devices,
            num_nodes=training_config.trainer.num_nodes,
            accelerator=training_config.trainer.accelerator,
            strategy={
                "name": training_config.trainer.strategy.name,
                **training_config.trainer.strategy.kwargs,
            },
            precision=training_config.trainer.precision,
        ),
        devices=training_config.trainer.devices,
        num_nodes=training_config.trainer.num_nodes,
        precision=training_config.trainer.precision,
        logger=lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=ckpt_save_dir,
            name="lightning_logs",
            version="latest",
        ),
        callbacks=callbacks,
        min_steps=training_config.trainer.min_steps,
        max_steps=training_config.trainer.max_steps,
        min_epochs=training_config.trainer.min_epochs,
        max_epochs=training_config.trainer.max_epochs,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=training_config.trainer.num_sanity_val_steps,
        log_every_n_steps=1,
        accumulate_grad_batches=training_config.trainer.accumulate_grad_batches,
        gradient_clip_val=training_config.trainer.gradient_clip_val,
        use_distributed_sampler=False,
    )
    trainer.fit(model=pl_module, ckpt_path=resume_from)


@click.group()
def main():
    pass


@main.command(name="acoustic", help="Train acoustic model.")
@click.option(
    "--config", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the configuration file."
)
@click.option(
    "--override", multiple=True,
    type=click.STRING, required=False,
    help="Override configuration values in dotlist format."
)
@click.option(
    "--work-dir", type=click.Path(
        dir_okay=True, file_okay=False, path_type=pathlib.Path
    ),
    required=False, default=pathlib.Path(__file__).parent.parent / "experiments",
    show_default=True,
    help="Path to the working directory. The experiment subdirectory will be created here."
)
@click.option(
    "--exp-name", type=click.STRING,
    required=True,
    help="Experiment name. Checkpoints and logs will be saved in subdirectory with this name."
)
@click.option(
    "--resume-from", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=False,
    help="Resume training from this checkpoint."
)
def _train_acoustic_model_cli(
        config: pathlib.Path, override: list[str],
        exp_name: str, work_dir: pathlib.Path,
        resume_from: pathlib.Path,
):
    config = _load_and_log_config(config, scope=ConfigurationScope.ACOUSTIC, overrides=override)
    ckpt_save_dir = work_dir / exp_name
    from training.acoustic_module import AcousticLightningModule
    train_model(
        config=config, pl_module_cls=AcousticLightningModule,
        ckpt_save_dir=ckpt_save_dir, resume_from=resume_from
    )


if __name__ == '__main__':
    main()
