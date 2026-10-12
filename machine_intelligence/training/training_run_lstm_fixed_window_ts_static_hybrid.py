import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from machine_intelligence.utilities.fixed_window_pred_dataset import (
    FixedWindowPredictionTsStaticHybridDataset,
)
from machine_intelligence.utilities.data_handling import load_parquet_table
from machine_intelligence.utilities.dataset_utils import (
    collate_fn_variable_length_batch_pad_hybrid_orthogonal_data,
)
from machine_intelligence.models.LSTMs import FixedWindowLSTM_TS_Static_Hybrid
from machine_intelligence.training.train_one_epoch_fixed_window_with_hybrid import train_one_epoch_fw
from machine_intelligence.validation.validate_training_fixed_window_hybrid import validate_fw

import torchmetrics
import wandb

wandb.require("core")


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

    force_cpu: bool = cfg.experiment.force_cpu

    if force_cpu:
        device: torch.device = torch.device("cpu")
        logger.info(f"Forcing training device/hardware to: {device}")
    else:
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Training device/hardware = {device} ")

    # dataset
    train_set_path: Path = Path(cfg.data.paths.df_train)
    val_set_path: Path = Path(cfg.data.paths.df_val)
    test_set_path: Path = Path(cfg.data.paths.df_test)

    # the static ds doesn't need a split, since the information is retrieved via the patient id
    static_variables_path: Path = Path(cfg.data.paths.static)

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

    learning_rate = cfg.optimizer.params.learning_rate
    # num_features = cfg.model.net.number_features
    num_hidden_units = cfg.model.net.number_hidden_units
    num_layers = cfg.model.net.number_layers

    # --- config end

    # wandb setup

    tracked_run = cfg.experiment.tracked_run

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
            # save_code=True
        )
        # this enables code saving make sure you don't save data by accident as this might flood the free storage plan
    else:
        logger.info(f"Not tracking this run! {tracked_run = }")

    df_train = load_parquet_table(
        parquet_path=train_set_path, verbose=True, backend="numpy_nullable"
    )

    df_val = load_parquet_table(parquet_path=val_set_path, verbose=True, backend="numpy_nullable")

    df_test = load_parquet_table(parquet_path=test_set_path, verbose=True, backend="numpy_nullable")

    df_static = load_parquet_table(
        parquet_path=static_variables_path, verbose=False, backend="numpy_nullable"
    )

    hybrid_var_path = Path("datasets/sepsis_type_b_cohort/hybrid/lab.parquet")

    df_hybrid = load_parquet_table(hybrid_var_path, backend="numpy_nullable")
    df_hybrid_test = df_hybrid.loc[:, ["patientunitstayid", "time_offset", "labresult", "labname"]]

    logger.info(f"{df_train.head()}")

    target = "sepsis_present"

    considered_features: list = OmegaConf.to_object(cfg.data.considered_feature_subset_training)

    train_dataset = FixedWindowPredictionTsStaticHybridDataset(
        dataframe_time_series=df_train,
        dataframe_hybrid=df_hybrid_test,
        dataframe_static=df_static,
        target=target,
        features=considered_features,
    )
    val_dataset = FixedWindowPredictionTsStaticHybridDataset(
        dataframe_time_series=df_val,
        dataframe_hybrid=df_hybrid_test,
        dataframe_static=df_static,
        target=target,
        features=considered_features,
    )
    test_dataset = FixedWindowPredictionTsStaticHybridDataset(
        dataframe_time_series=df_test,
        dataframe_hybrid=df_hybrid_test,
        dataframe_static=df_static,
        target=target,
        features=considered_features,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_variable_length_batch_pad_hybrid_orthogonal_data,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_variable_length_batch_pad_hybrid_orthogonal_data,
    )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_cats: int = len(df_hybrid_test["labname"].cat.categories)
    # embedding_size = (num_cats, min(50, num_cats + 1 // 2))
    embedding_size = (num_cats, 1)

    model = FixedWindowLSTM_TS_Static_Hybrid(
        num_features_ts=len(considered_features),
        embedding_size=embedding_size,
        num_features_static=12,
        hidden_size=num_hidden_units,
        num_layers=num_layers,
    )

    # Todo: Refactor class weight Berechnung wenn das mit dem stand geht
    df_patients_and_sepsis_status = df_train[
        ["sepsis_patient", "patientunitstayid"]
    ].drop_duplicates()

    label_classes = df_patients_and_sepsis_status["sepsis_patient"].value_counts().to_dict()
    total_number_samples: int = len(df_train["patientunitstayid"].unique().tolist())
    total_number_classes: int = len(label_classes)

    class_weights: list = [
        total_number_samples / (total_number_classes * label_classes[key]) for key in label_classes
    ]

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # performance metrics

    binary_accuracy_training = torchmetrics.classification.Accuracy(task="binary").to(device=device)
    binary_accuracy_validation = torchmetrics.classification.Accuracy(task="binary").to(
        device=device
    )

    performance_metrics: dict = {
        "acc_train": binary_accuracy_training,
        "acc_val": binary_accuracy_validation,
    }

    def train(model_to_train, optimizer_train, num_epochs, loader_train, loader_val, device_train):
        dict_log = {
            "train_acc_epoch": [],
            "val_acc_epoch": [],
            "train_loss_epoch": [],
            "val_loss_epoch": [],
        }

        progress_bar = tqdm(range(num_epochs), total=num_epochs, unit="Epoch")

        for epoch in progress_bar:  # noqa
            current_epoch: int = epoch + 1  # noqa
            msg: str = f"Epoch {current_epoch}/{num_epochs}"
            progress_bar.set_description(msg)

            (
                train_loss_epoch,
                train_acc_epoch,
            ) = train_one_epoch_fw(
                model_under_training=model_to_train,
                loss_func=loss_fn,
                optimizer_train=optimizer_train,
                loader_train=loader_train,
                device_training=device,
                perf_metrics=performance_metrics,
            )

            val_loss_epoch, val_acc_epoch = validate_fw(
                model_to_validate=model_to_train,
                loss_func=loss_fn,
                val_loader=loader_val,
                device_val=device,
                perf_metrics=performance_metrics,
            )

            dict_log["train_acc_epoch"].append(train_acc_epoch)
            dict_log["val_acc_epoch"].append(val_acc_epoch)
            dict_log["train_loss_epoch"].append(train_loss_epoch)
            dict_log["val_loss_epoch"].append(val_loss_epoch)

            if tracked_run:
                wandb.log(
                    {"train/acc": train_acc_epoch, "train/loss": train_loss_epoch}, step=epoch
                )
                wandb.log({"val/acc": val_acc_epoch, "val/loss": val_loss_epoch}, step=epoch)

        return dict_log

    history_dict = train(
        model_to_train=model,
        optimizer_train=optimizer,
        num_epochs=n_epochs,
        loader_train=train_data_loader,
        loader_val=val_data_loader,
        device_train=device,
    )

    wandb.run.log_code("machine_intelligence")

    logger.info("Training Summary:")
    for idx in range(n_epochs):
        ep_train_acc = history_dict["train_acc_epoch"][idx]
        ep_val_acc = history_dict["val_acc_epoch"][idx]

        ep_train_loss_acc = history_dict["train_loss_epoch"][idx]
        ep_val_loss_acc = history_dict["val_loss_epoch"][idx]

        logger.info(
            f"Epoch #{idx:3}: Accuracy [Training | Validation] {ep_train_acc :3.2%} - {ep_val_acc:3.2%}"
            f" Loss [Training | Validation] {ep_train_loss_acc:.3f} - {ep_val_loss_acc:.3f}"
        )

        wandb.finish()


if __name__ == "__main__":
    main()
