import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from machine_intelligence.utilities.data_handling import load_parquet_table
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from machine_intelligence.utilities.fixed_window_pred_dataset import (
    FixedWindowPredictionTsStaticDataset,
)
from machine_intelligence.models.LSTMs import FixedWindowLSTM_TS_Static
from machine_intelligence.training.train_one_epoch_fixed_window import train_one_epoch_fw
from machine_intelligence.validation.validate_training_fixed_window import validate_fw
from machine_intelligence.evaluation.test_performance import test_performance_fw


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

    all_columns: list = df_train.columns.tolist()
    feature_subset: list = OmegaConf.to_object(cfg.data.considered_feature_subset_training)

    aux_columns: list = OmegaConf.to_object(cfg.data.auxillary_columns)
    ignored_features: list = OmegaConf.to_object(cfg.data.ignored_columns)

    target: str = cfg.data.target_column
    # target = "sepsis_present"

    if len(feature_subset) == 0:
        considered_features = list(
            set(all_columns) - {target} - set(aux_columns) - set(ignored_features)
        )
    else:
        considered_features = list(
            set(feature_subset) - {target} - set(aux_columns) - set(ignored_features)
        )

    considered_features_static_cat = OmegaConf.to_object(cfg.data.considered_features_static_cat)
    considered_features_static_num = OmegaConf.to_object(cfg.data.considered_features_static_num)
    considered_features_static = [*considered_features_static_cat, *considered_features_static_num]

    logger.info(f"{all_columns = }")
    logger.info(f"{aux_columns = }")
    logger.info(f"{ignored_features = }")
    logger.info(f"{considered_features = }")
    logger.info(f"{target = }")

    sample_index_column: str = cfg.data.sample_idx_column

    # saving some metadata
    if tracked_run:
        # Note: These features include the "sepsis_present", it only used as a target var
        # wandb.log(data={"columns considered": considered_features}, step=None)
        wandb.config["columns_considered"] = considered_features

    train_dataset = FixedWindowPredictionTsStaticDataset(
        dataframe_time_series=df_train,
        dataframe_static=df_static,
        target=target,
        features=considered_features,
        features_static_cat=considered_features_static_cat,
        features_static_num=considered_features_static_num,
        sample_index_column_name=sample_index_column,
        auxillary_columns=aux_columns,
    )
    val_dataset = FixedWindowPredictionTsStaticDataset(
        dataframe_time_series=df_val,
        dataframe_static=df_static,
        target=target,
        features=considered_features,
        features_static_cat=considered_features_static_cat,
        features_static_num=considered_features_static_num,
        sample_index_column_name=sample_index_column,
        auxillary_columns=aux_columns,
    )
    test_dataset = FixedWindowPredictionTsStaticDataset(
        dataframe_time_series=df_test,
        dataframe_static=df_static,
        target=target,
        features=considered_features,
        features_static_cat=considered_features_static_cat,
        features_static_num=considered_features_static_num,
        sample_index_column_name=sample_index_column,
        auxillary_columns=aux_columns,
    )

    if debug_data := False:
        train_dataset, val_dataset, _ = random_split(train_dataset, [0.01, 0.01, 0.98])

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12
   )
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=12
    )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # we need to calculate the number of features with respect to the static data we use
    # this is due to the fact that we use pandas dummies, which sets up a column for each
    # distinct value of a categorical feature
    num_static_cat_features: int = len(
        pd.get_dummies(df_static.loc[:, considered_features_static_cat], dummy_na=True).columns
    )

    number_of_features_ts = len(considered_features)

    number_of_features_static = num_static_cat_features + len(considered_features_static_num)

    model = FixedWindowLSTM_TS_Static(
        num_features_ts=number_of_features_ts,
        num_features_static=number_of_features_static,
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

    logger.info(f"Class weights: {class_weights}")

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

        progress_bar = tqdm(range(1, num_epochs + 1), total=num_epochs, unit="Epoch")

        for epoch in progress_bar:  # noqa
            current_epoch: int = epoch  # noqa
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

            # if tracked_run:
            #     wandb.log(
            #         {"train/acc": train_acc_epoch, "train/loss": train_loss_epoch},
            #         step=epoch,
            #     )
            #
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
                    {"train/acc": train_acc_epoch, "train/loss": train_loss_epoch,
                     "val/acc": val_acc_epoch, "val/loss": val_loss_epoch},
                    step=epoch,
                )

        return dict_log

    history_dict = train(
        model_to_train=model,
        optimizer_train=optimizer,
        num_epochs=n_epochs,
        loader_train=train_data_loader,
        loader_val=val_data_loader,
        device_train=device,
    )

    if tracked_run:
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

    accuracy_test: torchmetrics.Metric = torchmetrics.classification.Accuracy(task="binary", average="macro")
    specificity_test: torchmetrics.Metric = torchmetrics.classification.Specificity(task="binary", average="macro")
    precision_test: torchmetrics.Metric = torchmetrics.classification.Precision(task="binary", average="macro")
    recall_test: torchmetrics.Metric = torchmetrics.classification.Recall(task="binary", average="macro")
    f1_score_test: torchmetrics.Metric = torchmetrics.classification.F1Score(task="binary")

    prc_rec_crv_test: torchmetrics.Metric = torchmetrics.classification.PrecisionRecallCurve(task="binary")
    confusion: torchmetrics.Metric = torchmetrics.ConfusionMatrix(task="binary")
    # Todo: balanced accuracy

    test_metrics: dict = {
        "acc": accuracy_test,
        "spec": specificity_test,
        "prc": precision_test,
        "rec": recall_test,
        "f1": f1_score_test,
        "prc_rec_crv": prc_rec_crv_test,
        "conf": confusion,
    }

    test_result, test_labels, test_preds, perf_report, _ = test_performance_fw(
        model_to_eval=model,
        dataloader=test_data_loader,
        device_test=device,
        perf_metrics=test_metrics,
        plot_conf_mat=True,
        plot_pr_curve=True,
    )


    wandb.finish()


if __name__ == "__main__":
    main()
