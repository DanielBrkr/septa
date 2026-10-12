import pandas as pd
from pathlib import Path
from loguru import logger
import hydra

from omegaconf import DictConfig, OmegaConf

from utils import data_handling


@logger.catch
@hydra.main(version_base=None, config_path="../conf", config_name="config_data.yaml")
def main(cfg: DictConfig):
    # general config portion
    hydra_log_path: str = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/run.log"
    )
    logger.add(f"{hydra_log_path}")

    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    debug_mode: bool = cfg.debug_mode

    # table specific configuration
    label_interval: str = cfg.data.interval
    data_filename: str = cfg.data.name

    inclusion_bound_lower_h: int = cfg.data.inclusion_bound_lower_h
    inclusion_bound_upper_h: int = cfg.data.inclusion_bound_upper_h

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    data_input_path: Path = data_root / Path(cfg.data.paths.sepsis_table)

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp

    parquet_data_path: Path = data_input_path
    data_output_dir: Path = Path(data_root) / Path(save_sub_dir) / Path(date)
    patient_ids_dir: Path = Path(data_root) / Path(cfg.data.directories.patient_ids)

    data_output_dir.mkdir(parents=True, exist_ok=True)
    # output_filename: str = "".join([data_filename + "_" + label_interval + "_" ])
    output_filename: str = f"{data_filename}_{label_interval}_{inclusion_bound_lower_h}h_{inclusion_bound_upper_h}h"

    data_root: Path = Path(cfg.directories.outputs_root)

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading file: {parquet_data_path}")

    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    # config end

    df_sepsis: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path, debug_mode=debug_mode, backend="numpy_nullable"
    )

    # df_sepsis["sepsis_offset"] = df_sepsis[
    #     ["labresultoffset", "abx_time", "samp_time"]
    # ].max(axis=1)
    #

    df_sepsis["sepsis_offset"] = df_sepsis["labresultoffset"]

    offsets_max_abs = abs(
        df_sepsis[["labresultoffset", "abx_time", "samp_time"]].max(axis=1)
    )
    offsets_min_abs = abs(
        df_sepsis[["labresultoffset", "abx_time", "samp_time"]].min(axis=1)
    )

    df_sepsis["time_delta"] = abs(offsets_max_abs - offsets_min_abs)

    df_sepsis_concise = df_sepsis[
        ["patientunitstayid", "sep3", "sepsis_offset", "time_delta"]
    ]

    # our bound for inclusion are, well, inclusive
    sepsis_inclusion_upper_bound = df_sepsis_concise["sepsis_offset"] <= pd.to_timedelta(
        inclusion_bound_upper_h, unit="hours"
    )

    sepsis_inclusion_lower_bound = df_sepsis_concise["sepsis_offset"] >= pd.to_timedelta(
        inclusion_bound_lower_h, unit="hours"
    )

    sepsis_exclusion_upper_bound = df_sepsis_concise["sepsis_offset"] > pd.to_timedelta(
        inclusion_bound_upper_h, unit="hours"
    )
    sepsis_exclusion_lower_bound = df_sepsis_concise["sepsis_offset"] < pd.to_timedelta(
        inclusion_bound_lower_h, unit="hours"
    )

    # df_sepsis_cohort_excluded_upper = df_sepsis_concise[sepsis_exclusion_upper_bound]

    df_sepsis_cohort_included = df_sepsis_concise[
        # df_sepsis_concise["sepsis_offset"] <= pd.to_timedelta(168, unit="hours")
        sepsis_inclusion_upper_bound & sepsis_inclusion_lower_bound
    ]

    df_sepsis_cohort_excluded = df_sepsis_concise[
        sepsis_exclusion_lower_bound ^ sepsis_exclusion_upper_bound
    ]

    logger.info(f"Sepsis cohort included stats: \n{df_sepsis_cohort_included.describe()}")

    logger.info(f"Sepsis cohort excluded stats: \n{df_sepsis_cohort_excluded.describe()}")

    ids_included_sepsis = df_sepsis_cohort_included["patientunitstayid"].unique().tolist()
    df_included_ids_sepsis: pd.DataFrame = pd.DataFrame(
        data=ids_included_sepsis, columns=["Included ID"]
    )

    ids_excluded_sepsis = df_sepsis_cohort_excluded["patientunitstayid"].unique().tolist()
    df_excluded_ids_sepsis: pd.DataFrame = pd.DataFrame(
        data=ids_excluded_sepsis, columns=["Excluded ID"]
    )
    patient_ids_dir.mkdir(exist_ok=True, parents=True)

    df_included_ids_sepsis.to_parquet(
        patient_ids_dir
        / Path(
            f"included_ids_sepsis_{label_interval}_{inclusion_bound_lower_h}h_{inclusion_bound_upper_h}h.parquet"
        ),
        index=False,
    )

    df_excluded_ids_sepsis.to_parquet(
        patient_ids_dir
        / Path(
            f"excluded_ids_sepsis_{label_interval}_{inclusion_bound_lower_h}h_{inclusion_bound_upper_h}h.parquet"
        ),
        index=False,
    )

    data_handling.save_parquet_table(
        df_in=df_sepsis, save_path=data_output_dir, filename=output_filename
    )

    logger.success("Sepsis labels have been filtered and saved!")


if __name__ == "__main__":
    main()
