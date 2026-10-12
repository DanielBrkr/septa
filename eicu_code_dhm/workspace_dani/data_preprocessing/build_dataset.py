from loguru import logger
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# from workspace_dani.data_preprocessing.utils.data_handling import (
from utils.data_handling import (
    load_parquet_table,
    save_parquet_table,
)

# from workspace_dani.data_preprocessing.utils.data_crafting import (
from utils.data_crafting import (
    attach_sepsis_labels_to_merged_tables,
)


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

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp
    data_output_dir: Path = (
            Path(data_root) / Path(save_sub_dir) / Path(cohort_name) / Path(date)
    )

    data_output_dir.mkdir(parents=True, exist_ok=True)

    merge_tables_path: Path = data_root / Path(cfg.data.paths.merged_timeseries_tables)
    sepsis_labels_path: Path = data_root / Path(cfg.data.paths.sepsis_labels)

    output_file_path: Path = data_output_dir

    logger.info(f"Paths and Directories {'-' * 100}")
    # logger.info(f"Loading files from: {data_input_dir}")

    logger.info(f"{cohort_name = }")
    logger.info(f"{data_root = }")
    logger.info(f"{merge_tables_path = }")
    logger.info(f"{sepsis_labels_path = }")
    logger.info(f"{data_output_dir = }")

    # config end

    df_merged_tables = load_parquet_table(
        parquet_path=merge_tables_path, verbose=False, debug_mode=False, backend="numpy_nullable"
    )

    df_sepsis_labels = load_parquet_table(parquet_path=sepsis_labels_path, verbose=False, backend="numpy_nullable")

    df_dataset_build = attach_sepsis_labels_to_merged_tables(
        df_merged_tabs=df_merged_tables, df_sepsis_labels=df_sepsis_labels
    )

    print(df_dataset_build.head())

    logger.warning(f"Build Dataset: saved table time offset stats: \n {df_dataset_build .time_offset.describe() = }")

    save_parquet_table(
        df_in=df_dataset_build,
        save_path=output_file_path,
        filename="dataset_ts",
    )


if __name__ == "__main__":
    main()
