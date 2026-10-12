import pandas as pd
from pathlib import Path
from loguru import logger
import hydra

from omegaconf import DictConfig, OmegaConf

from functools import reduce
from utils import data_handling


@logger.catch
@hydra.main(version_base=None, config_path="../conf", config_name="config_cohort.yaml")
def main(cfg: DictConfig):
    # general config portion
    hydra_log_path: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/run.log"
    logger.add(f"{hydra_log_path}")

    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    debug_mode: bool = cfg.debug_mode

    # table specific configuration
    data_filename: str = cfg.cohort.table.file_name
    table_name: str = cfg.cohort.table.table_name

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    data_input_dir: Path = data_root / Path(cfg.cohort.directories.input_data_dir)

    save_sub_dir = cfg.cohort.directories.data_save_dir

    date = cfg.metadata.timestamp

    parquet_data_path: Path = data_input_dir / Path(data_filename + ".parquet")
    data_output_dir: Path = Path(data_root) / Path(save_sub_dir) / Path(date)

    data_output_dir.mkdir(parents=True, exist_ok=True)
    output_filename: str = "".join([data_filename])

    sub_cohorts: list = cfg.cohort.sub_cohort
    data_root: Path = Path(cfg.directories.outputs_root)

    patient_ids_dir: Path = data_root / Path("patient_unit_stay_ids") / Path("cohorts")
    sepsis_to_exclude_path = data_root / Path(cfg.cohort.sepsis_to_exclude_path)

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading file: {parquet_data_path}")
    logger.info(f"{table_name = }")

    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    logger.info(f"{sub_cohorts= }")
    logger.info(f"{patient_ids_dir= }")
    logger.info(f"{sepsis_to_exclude_path= }")

    # config end

    df_table: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path, debug_mode=debug_mode
    )

    sub_cohorts_ids_paths: list[Path] = [
        Path(patient_ids_dir) / Path(config_value["id_path"])
        for config_key, config_value in sub_cohorts.items()
    ]
    logger.info(f"{sub_cohorts_ids_paths= }")

    sub_cohorts_ids_df_list: list = [
        data_handling.load_parquet_table(
            parquet_path=sub_cohort_path,
            debug_mode=False,
            # backend="numpy_nullable",
            verbose=False,
        )
        for sub_cohort_path in sub_cohorts_ids_paths
    ]

    df_sepsis_ids_to_exclude = data_handling.load_parquet_table(
        parquet_path=sepsis_to_exclude_path,
        debug_mode=False,
        # backend="numpy_nullable",
        verbose=False,
    )

    individual_cohorts_ids: list = [frame.iloc[:, 0].to_list() for frame in sub_cohorts_ids_df_list]

    ids_cohort_all = sorted(reduce(lambda x, y: set(x) & set(y), individual_cohorts_ids))

    sepsis_ids_to_exclude: list = df_sepsis_ids_to_exclude.iloc[:, 0].to_list()

    ids_cohort = sorted(set(ids_cohort_all) - set(sepsis_ids_to_exclude))

    df_table_cohort = df_table[df_table["patientunitstayid"].isin(ids_cohort)]

    data_handling.save_parquet_table(
        df_in=df_table_cohort, save_path=data_output_dir, filename=output_filename
    )

    logger.warning(f"Reduced to cohort: saved table stats: \n {df_table_cohort.describe() = }")

    logger.success(f" {data_filename} has been reduced to cohort!")


if __name__ == "__main__":
    main()
