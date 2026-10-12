import pandas as pd
from pathlib import Path
from loguru import logger
import hydra
import numpy as np

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
    cohort_name: str = cfg.data.cohort.sub_cohort.name
    data_filename: str = cfg.data.cohort.sub_cohort.source_table
    ids_name: str = cfg.data.cohort.sub_cohort.ids_name

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    data_input_dir: Path = data_root / Path(cfg.data.directories.input_data_dir)

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp

    parquet_data_path: Path = data_input_dir / Path(data_filename + ".parquet")
    data_output_dir: Path = (
        Path(data_root) / Path(save_sub_dir) / Path(cohort_name) / Path(date)
    )

    data_output_dir.mkdir(parents=True, exist_ok=True)
    output_filename: str = f"{ids_name}"

    sub_cohort: list = cfg.data.cohort.sub_cohort
    data_root: Path = Path(cfg.directories.outputs_root)

    patient_ids_dir: Path = data_root / Path("patient_unit_stay_ids") / Path("cohorts")

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading file: {parquet_data_path}")
    logger.info(f"{data_filename = }")

    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    logger.info(f"{sub_cohort= }")
    logger.info(f"{patient_ids_dir= }")

    # config end

    df_table: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path, debug_mode=debug_mode
    )

    if cohort_name == "admit_source":
        unit_admit_source_considered = ["Emergency Department", "Direct Admit"]

        df_admit_source_cohort = df_table[
            df_table["unitadmitsource"].isin(unit_admit_source_considered)
        ]

        df_admit_source_cohort_clean = df_admit_source_cohort.dropna(
            subset="unitadmitsource"
        )
        admit_source_value_counts = df_admit_source_cohort_clean[
            "unitadmitsource"
        ].value_counts()

        logger.debug(f"{admit_source_value_counts}")

        ids_included = df_admit_source_cohort["patientunitstayid"].unique().tolist()

        df_included_ids: pd.DataFrame = pd.DataFrame(
            data=ids_included, columns=["Included ID"]
        )

    elif cohort_name == "icu_type_a":
        icu_types_cohort_type_a = ["Med-Surg ICU", "MICU", "Cardiac ICU"]
        # icu_types_cohort_type_a = ["Med-Surg ICU", "MICU"]
        df_unit_type: pd.DataFrame = df_table[["patientunitstayid", "unittype"]]
        df_unit_type_a = df_unit_type[
            df_unit_type["unittype"].isin(icu_types_cohort_type_a)
        ]

        df_unit_type_a_value_counts = df_unit_type_a["unittype"].value_counts()

        logger.debug(f"{df_unit_type_a_value_counts = }")

        ids_included_type_a = df_unit_type_a["patientunitstayid"].unique().tolist()
        df_included_ids: pd.DataFrame = pd.DataFrame(
            data=ids_included_type_a, columns=["Included ID"]
        )

    elif cohort_name == "icu_type_b":
        icu_types_cohort_type_b = ["Cardiac ICU"]
        df_unit_type: pd.DataFrame = df_table[["patientunitstayid", "unittype"]]
        df_unit_type_b = df_unit_type[
            df_unit_type["unittype"].isin(icu_types_cohort_type_b)
        ]

        df_unit_type_b_value_counts = df_unit_type_b["unittype"].value_counts()
        logger.debug(f"{df_unit_type_b_value_counts = }")

        ids_included_type_b = df_unit_type_b["patientunitstayid"].unique().tolist()
        df_included_ids: pd.DataFrame = pd.DataFrame(
            data=ids_included_type_b, columns=["Included ID"]
        )

    elif cohort_name == "visit_number":
        unit_visit_number_upper_bound = 1

        df_visit_number_cohort = df_table[
            df_table["unitvisitnumber"] <= unit_visit_number_upper_bound
        ]

        df_visit_number_cohort_value_counts = df_visit_number_cohort[
            "unitvisitnumber"
        ].value_counts()
        logger.debug(f"{df_visit_number_cohort_value_counts= }")

        ids_included = df_visit_number_cohort["patientunitstayid"].unique().tolist()
        df_included_ids: pd.DataFrame = pd.DataFrame(
            data=ids_included, columns=["Included ID"]
        )

    elif cohort_name == "discharge_offset":
        df_unit_discharge_offsets = df_table[["patientunitstayid", "unitdischargeoffset"]]

        df_unit_discharge_offsets.insert(
            2,
            value=pd.to_timedelta(
                df_unit_discharge_offsets.loc[:, ("unitdischargeoffset")], unit="min"
            ),
            column="unitdischargeoffset_min",
        )

        df_unit_discharge_offsets.insert(
            3,
            value=df_unit_discharge_offsets.loc[:, "unitdischargeoffset_min"]
            / np.timedelta64(1, "h"),
            column="unitdischargeoffset_hours",
        )
        # bounds are inclusive
        unit_discharge_lower_bound: int = cfg.data.cohort.sub_cohort.discharge_lower_bound
        unit_discharge_upper_bound: int = cfg.data.cohort.sub_cohort.discharge_upper_bound

        lower_offset_condition = (
            df_unit_discharge_offsets["unitdischargeoffset_hours"]
            >= unit_discharge_lower_bound
        )
        upper_offset_condition = (
            df_unit_discharge_offsets["unitdischargeoffset_hours"]
            <= unit_discharge_upper_bound
        )

        df_unit_discharge_offsets_cohorts = df_unit_discharge_offsets.loc[
            lower_offset_condition & upper_offset_condition
        ]

        logger.debug(f"{df_unit_discharge_offsets_cohorts.describe() = }")

        ids_included = (
            df_unit_discharge_offsets_cohorts["patientunitstayid"].unique().tolist()
        )
        df_included_ids: pd.DataFrame = pd.DataFrame(
            data=ids_included, columns=["Included ID"]
        )

    elif cohort_name == "valid_apache_prediction":
        df_apache_pred_cohort = df_table[["patientunitstayid", "predictedicumortality"]]

        logger.debug(f"{df_apache_pred_cohort .describe() = }")

        ids_included = df_apache_pred_cohort["patientunitstayid"].unique().tolist()
        df_included_ids: pd.DataFrame = pd.DataFrame(
            data=ids_included, columns=["Included ID"]
        )

    else:
        df_included_ids = None

    data_handling.save_parquet_table(
        df_in=df_included_ids, save_path=data_output_dir, filename=output_filename
    )


if __name__ == "__main__":
    main()
