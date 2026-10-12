# Important Note!: Use this script in conjunction with an env that has been set up with
# "requirements_autoviz.txt", this is due to dependency clashes!

import os
import sys

import pandas as pd
from pathlib import Path
from loguru import logger
import hydra

from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf

# fixes import errors related to workspace_dani
sys.path.insert(1, os.getcwd())

from workspace_dani.data_preprocessing.utils.data_quality import (
    get_parameter_coverage_ratio_df,
    categorical_feature_frequency,
)

from workspace_dani.data_preprocessing.utils import data_handling


@logger.catch
@hydra.main(version_base=None, config_path="../conf", config_name="config_data.yaml")
def main(cfg: DictConfig):
    #  general config portion
    hydra_log_path: str = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/run.log"
    )
    logger.add(f"{hydra_log_path}")

    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    debug_mode: bool = cfg.debug_mode
    draft_mode: bool = cfg.draft_mode
    data_quality_stage: str = cfg.data.which_stage

    # overall data directory
    data_dir_path = Path(cfg.directories.data)

    # table specific configuration

    data_filename: str = cfg.data.dataset.table_name

    data_root: Path = Path(cfg.directories.outputs_root)

    data_outputs_root: str = cfg.directories.outputs_root
    percentiles: list = cfg.data.settings.percentiles

    logger.info(f"{percentiles = }")

    parquet_data_path: Path = data_root / Path(cfg.data.dataset.paths.table_location)

    data_output_dir: Path = Path(data_outputs_root) / Path(
        cfg.data.directories.data_save_dir
    )

    output_filename: str = f"{data_filename}"

    output_table_file_path: Path = data_output_dir / Path(output_filename)

    output_table_file_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Data directory: {Path(data_dir_path).resolve()}")
    logger.info(f"Loading file: {parquet_data_path}")

    logger.info(f"{data_outputs_root = }")
    logger.info(f"{data_output_dir = }")
    logger.info(f"{output_table_file_path = }")

    # save path setup for individual files
    descriptive_stats_output_csv_path: Path = output_table_file_path / Path(
        data_filename + "_descriptive_stats.csv"
    )

    descriptive_stats_per_patient_output_csv_path: Path = output_table_file_path / Path(
        data_filename + "_cov_ratio_descr_stats_per_patient.csv"
    )

    df_cats_stats_csv_path: Path = output_table_file_path / Path(
        data_filename + "_categorical_data.csv"
    )

    # config end

    df_table: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path,
        debug_mode=debug_mode,  # backend="numpy_nullable"
    )

    # this is hack such that autoviz doesn't fail to an ambiguous boolean comparison due to the NA dtype
    # df: pd.DataFrame = df.replace(to_replace=[-1.0], value=np.nan)

    # the time delta column has been removed in this step, not yet useful
    # if time_delta_column_name != "":
    #        df = df.drop(time_delta_column_name, axis=1)

    all_columns: list = df_table.columns.to_list()
    per_patient_coverage_ratios_df = get_parameter_coverage_ratio_df(
        df_grouped=df_table.groupby("patientunitstayid"), parameters=all_columns
    )

    descriptive_stats: pd.DataFrame = df_table.describe(
        percentiles=percentiles, include="all"
    ).T
    descriptive_stats_per_patient: pd.DataFrame = per_patient_coverage_ratios_df.describe(
        include="all",
        percentiles=percentiles,
    ).T

    logger.info(f"{'-' * 100}")

    df_cats_stats: pd.DataFrame = categorical_feature_frequency(
        frame=df_table, table_name=data_filename
    )

    # CSV and parquet saving
    # Note: Parquet saving seemed pretty buggy for this kind of data
    if not draft_mode:
        logger.info("Saving data quality!")
        descriptive_stats.to_csv(
            # path_or_buf=descriptive_stats_output_csv_path, decimal=","
            path_or_buf=descriptive_stats_output_csv_path,
            decimal=".",
        )
        descriptive_stats_per_patient.to_csv(
            # path_or_buf=descriptive_stats_per_patient_output_csv_path, decimal=","
            path_or_buf=descriptive_stats_per_patient_output_csv_path,
            decimal=".",
        )
        # df_data_quality.to_csv(path_or_buf=data_quality_csv_path, decimal=",")
        # df_cats_stats.to_csv(path_or_buf=df_cats_stats_csv_path, decimal=",")
        df_cats_stats.to_csv(path_or_buf=df_cats_stats_csv_path, decimal=".")

    logger.debug(f"Descriptive Stats for entire table: {'-' * 100} \n{descriptive_stats}")
    logger.debug(
        f"Descriptive Stats patient unit stay id wise: {'-' * 100} \n"
        f"{descriptive_stats_per_patient}"
    )
    logger.debug(f"Categorical Data Statistics: {'-' * 100} \n {df_cats_stats}")

    logger.success("Data Quality assessment done!")


if __name__ == "__main__":
    tqdm.pandas()
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG", diagnose=True)

    main()
