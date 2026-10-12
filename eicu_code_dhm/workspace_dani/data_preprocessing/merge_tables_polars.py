import polars as pl
from loguru import logger
from pathlib import Path
import hydra
from functools import reduce
from omegaconf import DictConfig, OmegaConf


@logger.catch
@hydra.main(version_base=None, config_path="../conf", config_name="config_data.yaml")
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
    cohort_name: str = cfg.data.cohort.name

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp
    data_output_dir: Path = Path(data_root) / Path(save_sub_dir) / Path(cohort_name) / Path(date)

    data_output_dir.mkdir(parents=True, exist_ok=True)

    output_filename_cats: str = "".join([data_filename, "_cats", ".parquet"])
    output_file_cats_path: Path = data_output_dir / output_filename_cats

    output_filename_ts: str = "".join([data_filename, "_ts", ".parquet"])
    # output_file_ts_path: Path = data_output_dir / output_filename_ts
    output_file_ts_path: Path = data_output_dir

    categorical_tables: list = cfg.data.used_tables.categorical_tables
    timeseries_tables: list = cfg.data.used_tables.timeseries_tables
    quasi_timeseries_tables: list = cfg.data.used_tables.quasi_timeseries_tables

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading files from: {data_input_dir}")

    logger.info(f"{cohort_name = }")
    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    # config end

    all_tables_paths: list = list(data_input_dir.iterdir())

    categorical_tables_paths: list = [
        table_path for table_path in all_tables_paths if table_path.name in categorical_tables
    ]

    timeseries_tables_paths: list = [
        table_path for table_path in all_tables_paths if table_path.name in timeseries_tables
    ]

    quasi_timeseries_tables_paths: list = [
        table_path for table_path in all_tables_paths if table_path.name in quasi_timeseries_tables
    ]

    logger.info(f"{categorical_tables_paths = }")
    logger.info(f"{timeseries_tables_paths = }")
    logger.info(f"{quasi_timeseries_tables_paths = }")

    df_list_cats: list[pl.DataFrame] = [
        pl.scan_parquet(table) for table in categorical_tables_paths
    ]
    frames_to_merge_cats: [pl.DataFrame] = [frame for frame in df_list_cats]

    df_joined_cats: pl.DataFrame = reduce(
        lambda left, right: left.join(right, how="left", on="patientunitstayid"),
        frames_to_merge_cats,
    )

    logger.info("Starting to save the categorical parquet file, this might take a while ! ")

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

        df_list_quasi_ts: list[pl.DataFrame] = [
            pl.read_parquet(table, low_memory=reduce_memory_pressure)
            for table in quasi_timeseries_tables_paths  # noqa
        ]

        frames_to_merge_quasi_ts: [pl.DataFrame] = [frame for frame in df_list_quasi_ts]
        frames_to_merge_ts: [pl.DataFrame] = [frame for frame in df_list_ts]

        # conversion to timedelta
        # for frame in frames_to_merge_ts:
        #    frame["time_offset"] = pd.to_timedelta(frame["time_offset"], unit="min")

        # for frame in frames_to_merge_quasi_ts:
        #    frame["time_offset"] = pd.to_timedelta(frame["time_offset"], unit="min")

        # resampling ticks
        # resampled_frames_to_merge_ts = list()
        # for frame in frames_to_merge_ts:
        #    resampled_frame = resample_patient_timeseries(df_to_resample=frame)
        #    resampled_frames_to_merge_ts.append(resampled_frame)

        # merging tables
        df_joined_ts: pl.DataFrame = reduce(
            lambda left, right: left.join(
                right,
                on=["patientunitstayid", "time_offset"],
                how="full",
                join_nulls=True,
            ),
            # resampled_frames_to_merge_ts,
            frames_to_merge_ts,
        )

        print(df_joined_ts.head())

        logger.info("Starting to save the time series parquet file, this might take a while ! ")

        df_joined_ts.write_parquet(
            output_file_ts_path / Path(output_filename_ts),
            compression="zstd",
            compression_level=10,
            use_pyarrow=True,
        )  # noqa

        logger.success("Saving time series parquet file done!")


if __name__ == "__main__":
    main()
