from pathlib import Path

import hydra
import lightning as lit
import torch
import torch.nn as nn
import wandb
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from machine_intelligence.models.LSTM import FixWinLSTM
from machine_intelligence.utilities.data_handling import load_parquet_table
from machine_intelligence.data.SepsisPredDataset import SepsisPredictionDataset
from machine_intelligence.data.TransformsData import ZeroFill
from machine_intelligence.interpretability.shap_analysis import shap_analysis

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

    n_workers: int = cfg.experiment.n_workers
    force_cpu: bool = cfg.experiment.force_cpu

    if force_cpu:
        device: torch.device = torch.device("cpu")
        logger.info(f"Forcing training device/hardware to: {device}")
    else:
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.warning(f"Training device/hardware = {device} ")

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

    lit.seed_everything(42)
    # --- config end

    # wandb setup

    tracked_run: bool = cfg.experiment.tracked_run
    saving_model: bool = cfg.experiment.saving_model
    dev_run: bool = cfg.experiment.dev_run
    profiling_flag: bool = cfg.experiment.profiling
    run_test: bool = cfg.experiment.run_test

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
    logger.info(f"{considered_features_static = }")
    logger.info(f"{target = }")

    sample_index_column: str = cfg.data.sample_idx_column

    # static data dtype mapping
    dtype_mapping = {
        "gender": "category",
        "aids": "boolean",
        "hepaticfailure": "boolean",
        "lymphoma": "boolean",
        "metastaticcancer": "boolean",
        "leukemia": "boolean",
        "immunosuppression": "boolean",
        "diabetes": "boolean",
        "admissionweight": "float",
        "admissionheight": "float",
        "age": "Int64",

    }

    # Apply the data types to the DataFrame
    for col, dtype in dtype_mapping.items():
        df_static[col] = df_static[col].astype(dtype)

    df_static_train = df_static.loc[
        df_static["patientunitstayid"].isin(df_train["patientunitstayid"]),
        ["patientunitstayid", *considered_features_static],
    ]

    df_static_val = df_static.loc[
        df_static["patientunitstayid"].isin(df_val["patientunitstayid"]),
        ["patientunitstayid", *considered_features_static],
    ]

    df_static_test = df_static.loc[
        df_static["patientunitstayid"].isin(df_test["patientunitstayid"]),
        ["patientunitstayid", *considered_features_static],
    ]

    # scaling and one-hot encoding the static variables based on the results of rus
    oh_encoder = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
    oh_encoder.fit(df_static_train[considered_features_static_cat])

    static_ohe_train = oh_encoder.transform(df_static_train[considered_features_static_cat])
    static_ohe_val = oh_encoder.transform(df_static_val[considered_features_static_cat])
    static_ohe_test = oh_encoder.transform(df_static_test[considered_features_static_cat])

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    scaler.fit(X=df_static_train[considered_features_static_num])
    static_scaled_train = scaler.transform(df_static_train[considered_features_static_num])
    static_scaled_val = scaler.transform(df_static_val[considered_features_static_num])
    static_scaled_test = scaler.transform(df_static_test[considered_features_static_num])

    # df_static_train_processed = pd.DataFrame(
    #     data=static_ohe_train, columns=oh_encoder.get_feature_names_out()
    # )
    df_static_train_processed = df_static_train.copy()
    # df_static_train_processed["patientunitstayid"] = df_static_train["patientunitstayid"].to_numpy()
    df_static_train_processed[scaler.get_feature_names_out()] = static_scaled_train

    # df_static_val_processed = pd.DataFrame(
    #     data=static_ohe_val, columns=oh_encoder.get_feature_names_out()
    # )
    df_static_val_processed = df_static_val.copy()
    # df_static_val_processed["patientunitstayid"] = df_static_val["patientunitstayid"].to_numpy()
    df_static_val_processed[scaler.get_feature_names_out()] = static_scaled_val

    # df_static_test_processed = pd.DataFrame(
    #     data=static_ohe_test, columns=oh_encoder.get_feature_names_out()
    # )

    df_static_test_processed = df_static_test.copy()
    # df_static_test_processed["patientunitstayid"] = df_static_test["patientunitstayid"].to_numpy()
    df_static_test_processed[scaler.get_feature_names_out()] = static_scaled_test

    # saving some metadata
    if tracked_run:
        # Note: These features include the "sepsis_present", it only used as a target var
        wandb.config["columns_considered"] = considered_features

    transforms = ZeroFill()
    # transforms = SampleAndHold()

    train_dataset = SepsisPredictionDataset(
        dataframe_time_series=df_train,
        dataframe_static=df_static_train_processed,
        target=target,
        features=considered_features,
        features_static_cat=considered_features_static_cat,
        features_static_num=considered_features_static_num,
        sample_index_column_name=sample_index_column,
        auxillary_columns=aux_columns,
        transform=transforms
    )
    val_dataset = SepsisPredictionDataset(
        dataframe_time_series=df_val,
        dataframe_static=df_static_val_processed,
        target=target,
        features=considered_features,
        features_static_cat=considered_features_static_cat,
        features_static_num=considered_features_static_num,
        sample_index_column_name=sample_index_column,
        auxillary_columns=aux_columns,
        transform=transforms
    )
    test_dataset = SepsisPredictionDataset(
        dataframe_time_series=df_test,
        dataframe_static=df_static_test_processed,
        target=target,
        features=considered_features,
        features_static_cat=considered_features_static_cat,
        features_static_num=considered_features_static_num,
        sample_index_column_name=sample_index_column,
        auxillary_columns=aux_columns,
        transform=transforms
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=True,
        num_workers=n_workers,
        # prefetch_factor=2,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        # pin_memory=True,
        num_workers=n_workers,
        persistent_workers=True,
        # prefetch_factor=3,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        # pin_memory=True,
        num_workers=n_workers,
        persistent_workers=True,
        # prefetch_factor=3,
    )

    number_of_features_ts = len(considered_features)

    number_of_features_static = len(df_static_train_processed.columns) - 1  # minus the aux column patientunitstayid

    logger.info(f"{number_of_features_ts=}, {number_of_features_static=}")

    # Todo: Refactor class weight Berechnung wenn das mit dem stand geht
    df_patients_and_sepsis_status = df_train[
        ["sepsis_sample", "sepsis_sample_shifted", "patientunitstayid", "sample_idx_6_h"]
    ]

    label_classes = (
        df_patients_and_sepsis_status.drop_duplicates(["sepsis_sample_shifted", "sample_idx_6_h"])[
            "sepsis_sample_shifted"
        ]
        .value_counts()
        .to_dict()
    )
    total_number_samples: int = len(df_train["sample_idx_6_h"].unique())
    total_number_classes: int = len(label_classes)

    class_weights: list = [
        total_number_samples / (total_number_classes * label_classes[key]) for key in label_classes
    ]
    logger.info(f"Class weights: {class_weights}")

    bce_class_weight = torch.tensor(class_weights[1] / class_weights[0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=bce_class_weight).to(device=device)

    model = FixWinLSTM(
        num_features_ts=number_of_features_ts,
        num_features_static=number_of_features_static,
        hidden_size=num_hidden_units,
        num_layers=num_layers,
        num_ts_output=8,
        num_static_output=2,
        loss_fn=loss_fn,
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        use_static_vars=True,
    )

    if profiling_flag:
        profiling = "simple"
    else:
        profiling = None

    if dev_run:
        train_batch_limit = 10
        val_batch_limit = 2
    else:
        train_batch_limit = None
        val_batch_limit = None

    trainer = lit.Trainer(
        # overfit_batches=2,
        fast_dev_run=False,
        # overfit_batches=1,  # debugging use the same batch over and over
        accelerator="auto",
        min_epochs=1,
        max_epochs=n_epochs,
        callbacks=[RichProgressBar(), LearningRateMonitor("epoch")],
        logger=[csv_logger, wandb_logger],
        limit_train_batches=train_batch_limit,
        limit_val_batches=val_batch_limit,
        num_sanity_val_steps=2,
        profiler=profiling,
    )

    trainer.fit(model=model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    if run_test:
        test_result = trainer.test(model=model, dataloaders=test_data_loader, verbose=True)

    batch_explainer_train = next(iter(train_data_loader))

    batch_explainer_test = next(iter(test_data_loader))

    # if dev_run:
    #     num_samples_attribution = 2
    #     logger.info(f"{num_samples_attribution =  }")
    # else:
    #     num_samples_attribution = batch_size

    # determine_temporal_and_static_attribution(
    #     model=model,
    #     batch_to_explain=batch_explainer_test,
    #     num_samples=num_samples_attribution,
    #     considered_features=considered_features,
    #     considered_features_static=oh_encoder.get_feature_names_out().tolist()
    # )

    num_total_features = number_of_features_ts + number_of_features_static

    # considered_features_all = [*considered_features, *oh_encoder.get_feature_names_out().tolist()]
    # considered_features_static_ohe = [*oh_encoder.get_feature_names_out().tolist(), *considered_features_static_num]

    model.eval()
    model.to("cpu")
    batch_explainer_test[0].to("cpu")
    batch_explainer_test[1].to("cpu")
    shap_analysis(model=model, batch_train=batch_explainer_train, batch_test=batch_explainer_test,
                  num_features=num_total_features, feature_names_ts=considered_features,
                  # feature_names_static=[*oh_encoder.get_feature_names_out().tolist(), *considered_features_static_num])
                  feature_names_static=[*considered_features_static_cat, *considered_features_static_num])

    if tracked_run:
        wandb.run.log_code("machine_intelligence")

    wandb.finish()


if __name__ == "__main__":
    main()
