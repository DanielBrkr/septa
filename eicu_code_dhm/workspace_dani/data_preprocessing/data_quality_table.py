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

from autoviz import data_cleaning_suggestions

# fixes import errors related to workspace_dani
sys.path.insert(1, os.getcwd())

from workspace_dani.data_preprocessing.utils.data_quality import (
    get_parameter_coverage_ratio_df,
    categorical_feature_frequency,
    count_specific_value_in_columns,
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
    # Note: We are getting some  information about the parquet table from the
    # plotting configuration which results in the "data.table.plot.table" structure
    data_filename: str = cfg.data.table.file_name
    file_suffix: str = cfg.data.table.file_suffix
    na_equivalent_value = cfg.data.table.metadata.na_equivalent_value

    logger.info(f"Considering following value as {na_equivalent_value = }")

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    # data_input_dir: Path = data_root / Path(cfg.data.directories.input_data_dir)

    # second pass i.e. post cleanup data quality assessment
    second_pass_indicator: str = cfg.data.which_stage

    if second_pass_indicator == "pre_cleanup":
        data_input_dir: Path = data_root / Path(
            cfg.data.directories.pre_clean_up_input_data_dir
        )
    elif second_pass_indicator == "post_cleanup":
        data_input_dir: Path = data_root / Path(
            cfg.data.directories.post_clean_up_input_data_dir
        )
    else:
        logger.error(
            f"There's something wrong with the pre or post cleanup indicator,"
            f" check the config! {second_pass_indicator}"
        )

    data_procedure_type: str = cfg.data.name
    data_outputs_root: str = cfg.directories.outputs_root
    percentiles: list = cfg.data.settings.percentiles

    logger.info(f"{percentiles = }")

    save_dir_modifier = cfg.data.directories.data_save_dir
    # parquet_data_path: Path = Path(cfg.data.table.paths.table_location)
    parquet_data_path: Path = data_input_dir / Path(data_filename + ".parquet")

    # if data_quality_stage == "dataset":
    #    dataset_time_series_path: Path = cfg.data.paths.dataset_time_series
    #    dataset_static_path: Path = cfg.data.paths.dataset_static
    #

    data_output_dir: Path = (
        Path(data_outputs_root) / Path(data_procedure_type) / Path(save_dir_modifier)
    )
    date = cfg.metadata.timestamp

    if file_suffix != "":
        # output_filename: str = "".join(
        #     [pre_or_post_data_cleanup, "_", data_filename, "_", file_suffix]
        # )
        output_filename: str = f"{data_quality_stage}_{data_filename}"
    else:
        output_filename: str = f"{data_quality_stage}_{data_filename}"

    output_table_file_path: Path = data_output_dir / Path(date) / Path(output_filename)

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
        data_filename + "_descriptive_stats_per_patient.csv"
    )

    na_equivalent_value_csv_path: Path = output_table_file_path / Path(
        data_filename + "_NA_equivalent_values.csv"
    )

    data_quality_csv_path: Path = output_table_file_path / Path(
        data_filename + "_data_quality.csv"
    )
    df_cats_stats_csv_path: Path = output_table_file_path / Path(
        data_filename + "_categorical_data.csv"
    )

    # config end

    df_table: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path, debug_mode=debug_mode
    )

    # this is hack such that autoviz doesn't fail to an ambiguous boolean comparison due to the NA dtype
    # df: pd.DataFrame = df.replace(to_replace=[-1.0], value=np.nan)

    time_delta_column_name = cfg.data.table.time_delta_column

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
        include="all"
    ).T

    if na_equivalent_value != "":
        logger.info(f"Replacing {na_equivalent_value = } with NA-values!")
        df_na_count = count_specific_value_in_columns(
            df=df_table, value=na_equivalent_value
        )
        logger.debug(
            f"Occasions of NaN-Equivalent Values {na_equivalent_value}:  \n {df_na_count}"
        )
    else:
        df_na_count = pd.DataFrame(
            data={str(data_filename): ["No NA equivalent values for this table"]}
        )

    logger.info(f"{'-' * 100}")
    try:
        df_data_quality = data_cleaning_suggestions(df_table)

    except TypeError as ex:
        logger.error(
            f"Probably autoviz error due to only NA cells in column, skipping this table {data_filename}"
            f" \n {ex}"
        )
        logger.warning("Trying to convert to autoviz compatible dtypes!")

        df_table = df_table.convert_dtypes(dtype_backend="pyarrow")
        logger.debug(
            f"Dtypes after dtype conversion due to autoviz error: \n {df_table.info()}"
        )
        df_data_quality = data_cleaning_suggestions(df_table)
        # df_data_quality = pd.DataFrame(data={str(data_filename): ["Autoviz bugs for this table"]})

    logger.info(f"{'-' * 100}")

    df_cats_stats: pd.DataFrame = categorical_feature_frequency(
        frame=df_table, table_name=data_filename
    )

    # CSV and parquet saving
    # Note: Parquet saving seemed pretty buggy for this kind of data
    if not draft_mode:
        logger.info("Saving data quality!")
        descriptive_stats.to_csv(
            path_or_buf=descriptive_stats_output_csv_path, decimal=","
        )
        descriptive_stats_per_patient.to_csv(
            path_or_buf=descriptive_stats_per_patient_output_csv_path, decimal=","
        )
        df_na_count.to_csv(path_or_buf=na_equivalent_value_csv_path, decimal=",")
        df_data_quality.to_csv(path_or_buf=data_quality_csv_path, decimal=",")
        df_cats_stats.to_csv(path_or_buf=df_cats_stats_csv_path, decimal=",")

    logger.debug(f"Descriptive Stats for entire table: {'-' * 100} \n{descriptive_stats}")
    logger.debug(
        f"Descriptive Stats patient unit stay id wise: {'-' * 100} \n"
        f"{descriptive_stats_per_patient}"
    )
    logger.debug(f"Data Cleaning Suggs: {'-' * 100} \n {df_data_quality}")
    logger.debug(f"Categorical Data Statistics: {'-' * 100} \n {df_cats_stats}")

    logger.success("Data Quality assessment done!")


if __name__ == "__main__":
    tqdm.pandas()
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG", diagnose=True)

    main()
