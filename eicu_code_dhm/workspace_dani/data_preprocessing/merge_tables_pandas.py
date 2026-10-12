import pandas as pd
import polars as pl
from loguru import logger
from pathlib import Path
import hydra
from functools import reduce
from omegaconf import DictConfig, OmegaConf

# from workspace_dani.data_preprocessing.utils.data_handling import save_parquet_table
from utils.data_handling import save_parquet_table

# from workspace_dani.data_preprocessing.utils.data_cleaning import (
from utils.data_cleaning import (
    merge_ts_tables,
)


# @logger.catch
@hydra.main(version_base=None, config_path="../conf", config_name="config_data.yaml")
def main(cfg: DictConfig):
    #  general config portion
    hydra_log_path: str = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/run.log"
    )
    logger.add(f"{hydra_log_path}")

    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    debug_mode: bool = cfg.debug_mode

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    data_input_dir: Path = data_root / Path(cfg.data.directories.input_data_dir)
    #    data_consolidated_input_dir: Path = data_root / Path(cfg.data.directories.input_consolidated_data_dir)
    data_filename: str = cfg.data.name
    cohort_name: str = cfg.data.cohort.name

    save_sub_dir = cfg.data.directories.data_save_dir
    save_sub_dir_static = cfg.data.directories.data_save_dir_static_data

    date = cfg.metadata.timestamp
    data_output_dir: Path = (
            Path(data_root) / Path(save_sub_dir) / Path(cohort_name) / Path(date)
    )
    data_output_dir_static: Path = (
            Path(data_root) / Path(save_sub_dir_static) / Path(cohort_name) / Path(date)
    )

    data_output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir_static.mkdir(parents=True, exist_ok=True)

    # output_filename_cats: str = "".join(["merged_tables", "_static", ".parquet"])
    output_filename_cats: str = "dataset_static.parquet"
    output_file_cats_path: Path = data_output_dir_static / output_filename_cats

    # output_filename_ts: str = "".join([data_filename, "_ts", ".parquet"])
    # output_file_ts_path: Path = data_output_dir / output_filename_ts
    output_file_ts_path: Path = data_output_dir

    categorical_tables: list = cfg.data.used_tables.categorical_tables
    timeseries_tables: list = cfg.data.used_tables.timeseries_tables
    hybrid_timeseries_tables: list = cfg.data.used_tables.hybrid_timeseries_tables

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading files from: {data_input_dir}")

    logger.info(f"{cohort_name = }")
    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    # config end

    # lab_path = data_consolidated_input_dir / Path(cfg.data.used_tables.lab_consolidated)
    lab_path = Path(data_input_dir) / Path(cfg.data.used_tables.lab_consolidated)

    tables_paths: list = list(data_input_dir.iterdir())
    consolidated_table_path: list = [lab_path.resolve()]

    all_tables_paths: list = [*tables_paths, *consolidated_table_path]

    categorical_tables_paths: list = [
        table_path
        for table_path in all_tables_paths
        if table_path.name in categorical_tables
    ]

    timeseries_tables_paths: list = [
        table_path
        for table_path in all_tables_paths
        if table_path.name in timeseries_tables
    ]

    timeseries_tables_paths: list = [*timeseries_tables_paths, *consolidated_table_path]

    hybrid_timeseries_tables_paths: list = [
        table_path
        for table_path in all_tables_paths
        if table_path.name in hybrid_timeseries_tables
    ]

    logger.info(f"{categorical_tables_paths = }")
    logger.info(f"{timeseries_tables_paths = }")
    logger.info(f"{hybrid_timeseries_tables_paths = }")

    df_list_cats: list[pl.DataFrame] = [
        pl.scan_parquet(table) for table in categorical_tables_paths
    ]
    frames_to_merge_cats: [pl.DataFrame] = [frame for frame in df_list_cats]

    df_joined_cats: pl.DataFrame = reduce(
        lambda left, right: left.join(
            right, how="left", on="patientunitstayid", coalesce=True
        ),
        frames_to_merge_cats,
    )

    # Todo: Struktur und Sortierung von den Outputfiles stimmt noch nicht
    logger.info(
        "Starting to save the categorical parquet file, this might take a while ! "
    )
    df_joined_cats.sink_parquet(
        output_file_cats_path,
        compression="zstd",  # noqa
        no_optimization=True,
    )  # , compression_level=10)
    logger.info("Saving categorical parquet file done!")

    if save_ts_dataset := True is True:
        logger.info("Starting to save time parquet file, this might take a while ! ")
        reduce_memory_pressure: bool = True
        logger.info(f"{reduce_memory_pressure = }")

        df_list_ts: list[pl.DataFrame] = [
            pl.read_parquet(table, low_memory=reduce_memory_pressure)
            for table in timeseries_tables_paths  # noqa
        ]

        df_list_hybrid_ts: list[pl.DataFrame] = [
            pl.read_parquet(table, low_memory=reduce_memory_pressure)
            for table in hybrid_timeseries_tables_paths  # noqa
        ]

        if use_pandas_for_merge := True:
            # Note!: Polars merge doesn't work as expected! It's just left in for documentation purposes for me!
            frames_to_merge_ts: [pd.DataFrame] = [
                frame.to_pandas() for frame in df_list_ts
            ]
            frames_to_merge_hybrid_ts: [pd.DataFrame] = [
                frame.to_pandas() for frame in df_list_hybrid_ts
            ]
            for df in frames_to_merge_ts:
                df["time_offset"] = pd.to_timedelta(df["time_offset"], unit='m')
        else:
            frames_to_merge_ts: [pl.DataFrame] = [frame for frame in df_list_ts]

        # merging tables
        if isinstance(frames_to_merge_ts[0], pl.DataFrame):
            df_joined_ts: pl.DataFrame = reduce(
                lambda left, right: left.join(
                    right,
                    on=["patientunitstayid", "time_offset"],
                    how="full",
                    join_nulls=True,
                ),
                frames_to_merge_ts,
            )
        else:
            all_tables_to_merge: list[pd.DataFrame] = [
                *frames_to_merge_ts,
                *frames_to_merge_hybrid_ts,  # noqa
            ]

            df_joined_ts = merge_ts_tables(dfs_to_merge=all_tables_to_merge)

        # print(df_joined_ts.head())
        # print(df_joined_ts.info())

        logger.info(
            "Starting to save the time series parquet file, this might take a while ! "
        )

        if isinstance(df_joined_ts, pd.DataFrame):
            save_parquet_table(
                df_in=df_joined_ts,
                save_path=output_file_ts_path,
                filename="merged_tables_ts",
            )
        elif isinstance(df_joined_ts, pl.DataFrame):
            df_joined_ts.write_parquet(
                output_file_ts_path,
                compression="zstd",
                compression_level=10,
                use_pyarrow=True,
            )  # noqa
        else:
            logger.error(
                f"Unable to save the time parquet file from dataframe with type {type(df_joined_ts)}"
            )


        logger.warning(f"Merge Tables: saved table stats: \n {df_joined_ts.time_offset.describe() = }")
        logger.success("Saving time series parquet file done!")


if __name__ == "__main__":
    main()
