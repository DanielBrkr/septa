import pandas as pd
import polars as pl
from loguru import logger
from pathlib import Path
import time
import sys
import hydra
from functools import reduce
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm, trange
import dask.dataframe as dd

import_path = Path.cwd() / Path("data_preprocessing") / Path("utils")

logger.info(f"Import path: {sys.path}")

# sys.path.insert(1, str(import_path))

from workspace_dani.data_preprocessing.utils import data_handling


@logger.catch
@hydra.main(version_base=None, config_path="../../conf", config_name="config_data.yaml")
def main(cfg: DictConfig):
    #  general config portion
    hydra_log_path: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/run.log"
    logger.add(f"{hydra_log_path}")

    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    debug_mode: bool = cfg.debug_mode

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    data_input_dir: Path = data_root / Path(cfg.data.directories.input_data_dir)
    data_filename: str = cfg.data.name

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp
    data_output_dir: Path = Path(data_root) / Path(save_sub_dir) / Path(date)

    data_output_dir.mkdir(parents=True, exist_ok=True)

    output_filename_cats: str = "".join([data_filename, "_cats", ".parquet"])
    output_file_cats_path: Path = data_output_dir / output_filename_cats

    output_filename_ts: str = "".join([data_filename, "_ts", ".parquet"])
    output_file_ts_path: Path = data_output_dir / output_filename_ts

    categorical_tables: list = cfg.data.settings.categorical_tables

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading files from: {data_input_dir}")

    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    # config end

    all_tables_paths: list = list(data_input_dir.iterdir())

    categorical_tables_paths: list = [
        table_path for table_path in all_tables_paths if table_path.name in categorical_tables
    ]

    timeseries_tables_paths: list = [
        table_path for table_path in all_tables_paths if table_path.name not in categorical_tables
    ]

    logger.info(f"{categorical_tables_paths = }")
    logger.info(f"{timeseries_tables_paths = }")

    # df_list_cats: list[pl.DataFrame] = [pl.scan_parquet(table) for table in categorical_tables_paths]
    # frames_to_merge_cats: [pl.DataFrame] = [frame for frame in df_list_cats]

    # Todo: In Categorical und Time Series Data aufsplitten
    # df_concat_cats: pl.DataFrame = pl.concat([*frames_to_merge_cats], how="diagonal_relaxed")
    # Todo: this is just for testing
    # df_joined_cats: pl.DataFrame = frames_to_merge_cats[0].join(frames_to_merge_cats[1], on="patientunitstayid", how="left")
    # df_merged_at_once = reduce(lambda left, right: pd.merge(left, right, on="patientunitstayid"), df_list)
    # df_joined_cats: pl.DataFrame = reduce(lambda left, right: left.join(right, how="left", on="patientunitstayid"),
    #                                      frames_to_merge_cats)
    # df_joined_cats: pl.DataFrame = frames_to_merge_cats[0].join(frames_to_merge_cats[1], how="left", on="patientunitstayid")

    # Todo: Struktur und Sortierung von den Outputfiles stimmt noch nicht
    # logger.info("Starting to save the categorical parquet file, this might take a while ! ")
    # df_concat_cats.sink_parquet(output_file_cats_path, compression="zstd")  # , compression_level=10)  # noqa
    # df_joined_cats.sink_parquet(output_file_cats_path, compression="zstd",
    #                            no_optimization=True)  # , compression_level=10)  # noqa
    # logger.info("Saving categorical parquet file done!")

    # reduce_memory_pressure: bool = True
    # logger.info(f"{reduce_memory_pressure =}")

    considered_ts_tables_path = timeseries_tables_paths[0:3]
    logger.info(f"{considered_ts_tables_path = }")

    df_list_ts: list[pd.DataFrame] = [
        dd.from_pandas(data_handling.load_parquet_table(parquet_path=table), npartitions=10)
        for table in considered_ts_tables_path
    ]
    # frames_to_merge_ts: [pd.DataFrame] = [frame for frame in df_list_ts]

    # df_list_ts: list[dd.DataFrame] = [dd.read_parquet(path=table, dtype_backend="pyarrow", ignore_metadata_file=True) for table in considered_ts_tables_path]

    # Todo: In Categorical und Time Series Data aufsplitten
    # df_concat_ts: pl.DataFrame = pl.concat([*frames_to_merge_ts], how="diagonal_relaxed")
    # df_merged_at_once = reduce(lambda left, right: pd.merge(left, right, on="patientunitstayid"), tqdm(df_list_ts))
    # df_merged_at_once = pd.merge(left=df_list_ts[0].head(n=1000), right=df_list_ts[1].head(n=1000), how="left", on="patientunitstayid")
    # df_merged_at_once = pd.merge(left=df_list_ts[0], right=df_list_ts[2], how="left", on="patientunitstayid")
    df_merged_at_once = dd.merge(
        left=df_list_ts[0], right=df_list_ts[2], how="left", on="patientunitstayid"
    )

    # Todo: Struktur und Sortierung von den Outputfiles stimmt noch nicht - Inner Join?
    logger.info("Starting to save the time series parquet file, this might take a while ! ")
    df_merged_at_once.to_parquet(path=output_file_ts_path)
    # df_concat_ts.sink_parquet(output_file_ts_path, compression="zstd", compression_level=None)  # noqa
    #    df_joined_ts.sink_parquet(output_file_ts_path, compression="zstd", compression_level=None)  # noqa
    logger.success("Saving time series parquet file done!")


if __name__ == "__main__":
    main()
