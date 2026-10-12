import torch
import torch.nn as nn
from loguru import logger
from data_preprocessing.utils.data_handling import load_parquet_table
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from playground.fixed_window_pred_dataset import FixedWindowPredictionDataset
from playground.deep_learning.fixed_window.LSTM_fixed_window import ShallowRegressionLSTM
from training_one_epoch import train_one_epoch
from training_validate import validate

import torchmetrics


@hydra.main(version_base=None, config_path="../../../configs", config_name="run_lstm")
def main(cfg: DictConfig) -> None:
    # --- config

    hydra_config: dict = hydra.core.hydra_config.HydraConfig.get()
    hydra_cwd: str = hydra_config['runtime']['cwd']
    hydra_output_dir: str = hydra_config['runtime']['output_dir']

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
    dataset_ts_path: Path = Path(cfg.data.paths.dataset_path)
    debug_dataset: bool = cfg.experiment.debug_dataset

    logger.debug(f"{dataset_ts_path = }")
    logger.debug(f"{debug_dataset = }")

    # training parameters

    n_epochs = cfg.params.n_epochs

    if device.type != "cuda":
        batch_size = cfg.params.batch_size_cpu
    else:
        batch_size = cfg.params.batch_size_gpu  # or multiples of 32

    learning_rate = cfg.optimizer.params.learning_rate
    num_hidden_units = cfg.model.net.number_hidden_units
    num_features = cfg.model.net.number_features

    # --- config end

    df_dataset_ts = load_parquet_table(parquet_path=dataset_ts_path, verbose=False)

    logger.info(f"{df_dataset_ts.head()}")

    patient_ids = df_dataset_ts["patientunitstayid"].unique().tolist()

    df_dataset_ts_reduced = df_dataset_ts.loc[:,
                            ["patientunitstayid", "time_offset", "heartrate", "sao2", "temperature", "st1",
                             "sepsis_present"]]

    df_dataset_ts_reduced = df_dataset_ts_reduced[df_dataset_ts_reduced["patientunitstayid"].isin(patient_ids)].copy()

    df_dataset_ts_sepsis = df_dataset_ts[
        df_dataset_ts["sepsis_patient"] == True]  # noqa: E712, supposed solution changes function

    patient_ids_sepsis = df_dataset_ts_sepsis["patientunitstayid"].unique().tolist()

    df_dataset_ts_no_sepsis = df_dataset_ts[
        df_dataset_ts["sepsis_patient"] == False]  # noqa: E712, supposed solution changes function

    patient_ids_no_sepsis = df_dataset_ts_no_sepsis["patientunitstayid"].unique().tolist()

    if debug_dataset:

        df_dataset_ts_debug = df_dataset_ts_reduced.loc[
            df_dataset_ts_reduced["patientunitstayid"].isin([*patient_ids_no_sepsis[0:10], *patient_ids_sepsis[0:10]])]

        df_train = df_dataset_ts_debug.loc[
            df_dataset_ts_debug["patientunitstayid"].isin([*patient_ids_no_sepsis[0:5], *patient_ids_sepsis[0:5]])]

        df_test = df_dataset_ts_debug.loc[
            df_dataset_ts_debug["patientunitstayid"].isin([*patient_ids_no_sepsis[0:5], *patient_ids_sepsis[0:5]])]
    else:

        df_train = df_dataset_ts_reduced.loc[df_dataset_ts_reduced["patientunitstayid"].isin(patient_ids[0:100])]
        df_test = df_dataset_ts_reduced.loc[df_dataset_ts_reduced["patientunitstayid"].isin(patient_ids[101:200])]

    logger.debug(f"{len(df_test) / len(df_train) = }")

    target = "sepsis_present"
    features = list(df_dataset_ts_reduced.columns.difference([target]))

    train_dataset = FixedWindowPredictionDataset(dataframe=df_train, target=target, features=features)
    test_dataset = FixedWindowPredictionDataset(dataframe=df_test, target=target, features=features)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  )

    model = ShallowRegressionLSTM(num_features=num_features, hidden_size=num_hidden_units)

    loss_fn = nn.CrossEntropyLoss().to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # performance metrics

    binary_accuracy_training = torchmetrics.classification.Accuracy(task="binary").to(device=device)
    binary_accuracy_validation = torchmetrics.classification.Accuracy(task="binary").to(device=device)

    performance_metrics: dict = {
        "acc_train": binary_accuracy_training,
        "acc_val": binary_accuracy_validation,
    }

    def train(
            model_to_train,
            optimizer_train,
            num_epochs,
            loader_train,
            loader_val,
            device_train: torch.device,
    ):
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
            ) = train_one_epoch(
                model_under_training=model_to_train,
                loss_func=loss_fn,
                optimizer_train=optimizer_train,
                loader_train=loader_train,
                device_training=device,
                perf_metrics=performance_metrics,
            )

            val_loss_epoch, val_acc_epoch = validate(
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

            # mlflow.log_metric("Loss/train", train_loss_epoch, step=epoch)
            # mlflow.log_metric("Loss/val", val_loss_epoch, step=epoch)
            # mlflow.log_metric("Acc/train", train_acc_epoch, step=epoch)
            # mlflow.log_metric("Acc/val", val_acc_epoch, step=epoch)

        return dict_log

    history_dict = train(
        model_to_train=model,
        optimizer_train=optimizer,
        num_epochs=n_epochs,
        loader_train=train_data_loader,
        # loader_val=val_data_loader,
        loader_val=test_data_loader,
        device_train=device,
    )

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


if __name__ == "__main__":
    main()
