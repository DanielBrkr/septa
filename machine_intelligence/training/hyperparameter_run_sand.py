import hydra
import lightning as lit
import torch
import torch.nn as nn
import wandb
import os
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from machine_intelligence.models.SAnD import SAnD
from machine_intelligence.data.SepsisPredDataModule import SepsisPredDataModule

wandb.require("core")


# Todo: https://www.kaggle.com/code/windsuzu/beyond-77-pytorch-lightning-ray-tune#%F0%9F%98%BD-Ray-Tune
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
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.warning(f"Training device/hardware = {device} ")

    dataset_name: str = cfg.data.name
    debug_dataset: bool = cfg.data.debug

    logger.debug(f"{debug_dataset = }")

    # metadata

    arch: str = cfg.model.architecture

    # training parameters
    n_epochs = cfg.params.n_epochs

    if device.type != "cuda":
        batch_size = cfg.params.batch_size_cpu
    else:
        batch_size = cfg.params.batch_size_gpu  # or multiples of 32

    learning_rate = cfg.optimizer.params.learning_rate

    lit.seed_everything(42)
    # --- config end

    # wandb setup

    tracked_run: bool = cfg.experiment.tracked_run
    saving_model: bool = cfg.experiment.saving_model
    dev_run: bool = cfg.experiment.dev_run

    if tracked_run:
        wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.experiment.experiment_name,
            # track hyperparameters and run metadata
            config={
                "epochs": n_epochs,
                "learning_rate": learning_rate,
                "architecture": arch,
                "dataset": dataset_name,
                "dataset_debug": debug_dataset,
                "experiment_config": hydra_experiment_setup,
            },
            save_code=True,
        )
        # # this enables code saving make sure you don't save data by accident as this might flood the free storage plan
        wandb_logger = WandbLogger(
            log_model=saving_model,
            save_dir="experiment_logs",
            project=cfg.experiment.experiment_name,
        )

        csv_logger = CSVLogger("experiment_logs", name=cfg.experiment.experiment_name)

    else:
        logger.info(f"Not tracking this run! {tracked_run = }")

    sepsis_pred = SepsisPredDataModule(cfg=cfg)


    loss_fn = nn.BCEWithLogitsLoss().to(device=device)

    arch_config = OmegaConf.to_container(cfg.model.net, resolve=True, throw_on_missing=True)
    optim_config = OmegaConf.to_container(cfg.optimizer.params, resolve=True, throw_on_missing=True)

    model = SAnD(
        input_features=len(sepsis_pred.considered_features),
        num_static_feat=len(sepsis_pred.considered_features_static),
        seq_len=72,
        num_classes=1,
        loss_fn=loss_fn,
        arch_config=arch_config,
        optimizer_name="Adam",
        optimizer_hparams={"lr": optim_config["learning_rate"], "weight_decay": optim_config["weight_decay"]},

    )

    if dev_run:
        train_batch_limit = 10
        val_batch_limit = 2
    else:
        train_batch_limit = None
        val_batch_limit = None

    trainer = lit.Trainer(
        fast_dev_run=False,
        accelerator="auto",
        min_epochs=1,
        # max_epochs=n_epochs,
        deterministic="warn",
        max_epochs=30,
        callbacks=[RichProgressBar(), LearningRateMonitor("epoch"),
                   EarlyStopping(monitor="val/loss_epoch", mode="min", patience=10, min_delta=0.001)],
        logger=[csv_logger, wandb_logger],
        limit_train_batches=train_batch_limit,
        limit_val_batches=val_batch_limit,
        num_sanity_val_steps=2,
    )

    trainer.fit(
        model=model,
        datamodule=sepsis_pred
    )

    if tracked_run:
        wandb.run.log_code("machine_intelligence")

    wandb.finish()


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(42)
    lit.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True)
    main()
