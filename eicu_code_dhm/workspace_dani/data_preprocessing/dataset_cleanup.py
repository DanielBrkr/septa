from loguru import logger
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm import tqdm
import yaml

# from workspace_dani.data_preprocessing.utils.data_handling import (
from utils.data_handling import (
    save_parquet_table,
    load_parquet_table,
)

tqdm.pandas()


@logger.catch
@hydra.main(version_base=None, config_path="../conf", config_name="config_data.yaml")
def main(cfg: DictConfig):
    #  general config portion
    hydra_log_path: str = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/run.log"
    )
    logger.add(f"{hydra_log_path}")

    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    # data_input_file_path: Path = data_root / Path(cfg.data.paths.merged_timeseries_tables)
    cohort_name: str = cfg.data.cohort.name

    # cleanup configuration
    variable_count_threshold: int = cfg.data.cleanup.variable_count_threshold
    aux_columns: list = OmegaConf.to_object(cfg.data.cleanup.aux_columns_to_drop)
    reduce_to_cohort_b_variables: bool = cfg.data.cleanup.reduce_to_cohort_b_columns

    logger.info(f"{variable_count_threshold = }")

    save_sub_dir = cfg.data.directories.data_save_dir
    date = cfg.metadata.timestamp
    data_output_dir: Path = (
        Path(data_root) / Path(save_sub_dir) / Path(cohort_name) / Path(date)
    )

    data_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path: Path = data_root / Path(cfg.data.paths.dataset_raw)

    output_file_path: Path = data_output_dir

    logger.info(f"Paths and Directories {'-' * 100}")
    # logger.info(f"Loading files from: {data_input_dir}")

    logger.info(f"{cohort_name = }")
    logger.info(f"{data_root = }")
    logger.info(f"{dataset_path= }")
    logger.info(f"{data_output_dir = }")

    # config end

    df_datset_raw = load_parquet_table(
        parquet_path=dataset_path, verbose=False, debug_mode=False
    )

    logger.info(f"{df_datset_raw.head() = }")

    ### discarding underpopulated columns

    # variable_value_counts = df_datset_raw.progress_apply(lambda x: x.value_counts())
    ser_variable_counts = df_datset_raw.count()
    ser_variable_counts.columns = "variable_counts"

    variables_to_discard = ser_variable_counts.loc[
        ser_variable_counts <= variable_count_threshold
    ]

    if cohort_name == "sepsis-type-a-cohort" and (use_b_cols:= False):
        # no recompute will ignore the features from cohort b
        cohort_b_columns = OmegaConf.to_object(
            cfg.data.cleanup.valid_columns_from_cohort_b
        )

        all_columns = df_datset_raw.columns.to_list()

        columns_to_drop = list(set(all_columns) - set(cohort_b_columns))

    else:
        columns_to_drop = [*variables_to_discard.index.tolist(), *aux_columns]
        df_ds_val_count_thresholded = df_datset_raw.drop(columns=columns_to_drop)

        yaml_file_path = Path("cohort_a_columns.yaml")
        pickle_file_path = output_file_path / Path("cohort_a_columns_pickle.pkl")

        columns = df_ds_val_count_thresholded.columns.tolist()
        pd.to_pickle(columns, pickle_file_path)

        with open(output_file_path / yaml_file_path, "w") as yaml_file:
            yaml.dump(columns, yaml_file)

    if cohort_name == "sepsis-type-b-cohort":
        df_ds_val_count_thresholded = df_datset_raw.drop(columns=columns_to_drop)
        yaml_file_path = Path("cohort_b_columns.yaml")

        columns = df_ds_val_count_thresholded.columns.tolist()
        pickle_file_path = output_file_path / Path("cohort_b_columns_pickle.pkl")
        pd.to_pickle(columns, pickle_file_path)

        with open(output_file_path / yaml_file_path, "w") as yaml_file:
            yaml.dump(columns, yaml_file)

    save_parquet_table(
        df_in=df_ds_val_count_thresholded,
        save_path=output_file_path,
        filename="dataset_ts_cleaned",
    )

    logger.info(f"{df_ds_val_count_thresholded.info() = }")

    logger.warning(
        f"Dataset Cleanup: saved table time offset stats: \n {df_ds_val_count_thresholded.time_offset.describe() = }"
    )


if __name__ == "__main__":
    main()
