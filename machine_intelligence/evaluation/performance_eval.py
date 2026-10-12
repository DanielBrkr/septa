from pathlib import Path

import hydra
import lightning as lit
import torch
import wandb

from machine_intelligence.data.SepsisPredDataModule import SepsisPredDataModule
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from datetime import datetime
import numpy as np
import os
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from machine_intelligence.models.GatedTransformer import GatedTransformer

wandb.require("core")

# torch.use_deterministic_algorithms(True)


@hydra.main(version_base=None, config_path="../../configs", config_name="run_lstm")
def main(cfg: DictConfig) -> None:
    # --- config


    hydra_config: dict = hydra.core.hydra_config.HydraConfig.get()
    hydra_cwd: str = hydra_config["runtime"]["cwd"]
    hydra_output_dir: str = hydra_config["runtime"]["output_dir"]
    hydra_experiment_setup = OmegaConf.to_yaml(cfg)

    logger.debug(f"Hydra working directory: {hydra_cwd}")
    logger.debug(f"Hydra output directory: {hydra_output_dir}")
    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    # general/compute

    n_workers: int = cfg.experiment.n_workers
    force_cpu: bool = cfg.experiment.force_cpu

    if force_cpu:
        device: torch.device = torch.device("cpu")
        logger.info(f"Forcing training device/hardware to: {device}")
    else:
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # if torch.cuda.is_available():
            # torch.cuda.manual_seed(42)
            # torch.cuda.manual_seed_all(42)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        logger.warning(f"Training device/hardware = {device} ")

    # dataset
    train_set_path: Path = Path(cfg.data.paths.df_train)
    val_set_path: Path = Path(cfg.data.paths.df_val)
    test_set_path: Path = Path(cfg.data.paths.df_test)

    dataset_name: str = cfg.data.name
    debug_dataset: bool = cfg.data.debug

    logger.debug(f"{debug_dataset = }")

    logger.debug(f"{train_set_path = }")
    logger.debug(f"{val_set_path = }")
    logger.debug(f"{test_set_path = }")

    # metadata

    arch: str = cfg.model.architecture

    # training parameters
    n_epochs = cfg.params.n_epochs

    if device.type != "cuda":
        batch_size = cfg.params.batch_size_cpu
    else:
        batch_size = cfg.params.batch_size_gpu  # or multiples of 32


    # --- config end

    # wandb setup


    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.experiment.experiment_name,
        # track hyperparameters and run metadata
    )
    # this enables code saving make sure you don't save data by accident as this might flood the free storage plan
    wandb_logger = WandbLogger(
        save_dir="experiment_logs",
        project=cfg.experiment.experiment_name,
    )

    csv_logger = CSVLogger("experiment_logs", name=cfg.experiment.experiment_name)


    cwd = Path.cwd()

    project_root = Path(cwd)

    current_date = datetime.now().strftime('%Y-%m-%d')
    plot_output_dir = project_root / 'plot_outputs' / current_date
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    # model_path = project_root / Path("experiment_logs/gtn-final/version_20/checkpoints/last.ckpt")
    # model_path = project_root / Path("experiment_logs/gtn-final/version_19/checkpoints/epoch=35-step=288.ckpt")
    model_hparams_path = project_root / Path("experiment_logs/gtn-final/version_15/hparams.yaml")
    model_path = project_root / Path("machine_intelligence/models/trained_models/gtn_final/checkpoints/last.ckpt")

    sepsis_pred = SepsisPredDataModule(cfg=cfg)

    arch_config = OmegaConf.to_container(cfg.model.net, resolve=True, throw_on_missing=True)
    optim_config = OmegaConf.to_container(cfg.optimizer.params, resolve=True, throw_on_missing=True)

    model = GatedTransformer.load_from_checkpoint(checkpoint_path=model_path)

    logger.info(f"Pre Test: {model.classifier_head[6].weight}")

    trainer = lit.Trainer(
        num_nodes=1,
        # deterministic=True,
        deterministic=True,
        fast_dev_run=False,
        devices=1,
        accelerator="auto",
        benchmark=False,
        logger=[csv_logger, wandb_logger],
        num_sanity_val_steps=0,

    )

    model.eval()

    logger.info(f"Pre Test: {model.classifier_head[6].weight =}")
    logger.info(f"Pre Test: {model.embedding_channel.weight =}")
    logger.info(f"Pre Test: {model.embedding_input.weight =}")

    test_result = trainer.test(
        model=model, datamodule=sepsis_pred, verbose=True, ckpt_path=Path("machine_intelligence/models/trained_models/gtn_final/checkpoints/last.ckpt")
    )

    logger.info(f"After Test: {model.classifier_head[6].weight =}")
    logger.info(f"After Test: {model.embedding_channel.weight =}")
    logger.info(f"After Test: {model.embedding_input.weight =}")

    print("hallo")


if __name__ == "__main__":

    os.environ['PYTHONHASHSEED'] = str(42)
    lit.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True)
    main()
