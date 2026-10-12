import pandas as pd
from pathlib import Path
from loguru import logger
import hydra

from omegaconf import DictConfig, OmegaConf

# fixes import errors related to workspace_dani
# sys.path.insert(1, str(Path.cwd()))

from utils import data_handling, data_cleaning


# Note: This script doesn't work in debug mode, i.e. use the complete tables. It seems like
# that the pyarrow metadata doesn't correspond correctly to the reduced tables


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

    # table specific configuration
    data_filename: str = cfg.data.table.file_name
    table_name: str = cfg.data.table.table_name
    na_equivalent_value = cfg.data.table.metadata.na_equivalent_value
    time_delta_column_name: str = cfg.data.table.time_offset_name

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)

    data_input_dir: Path = data_root / Path(cfg.data.directories.input_data_dir)

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp

    parquet_data_path: Path = data_input_dir / Path(data_filename + ".parquet")
    data_output_dir: Path = Path(data_root) / Path(save_sub_dir) / Path(date)

    data_output_dir.mkdir(parents=True, exist_ok=True)

    output_filename: str = "".join([data_filename])

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading file: {parquet_data_path}")
    logger.info(f"{table_name = }")
    logger.info(f"Considering following value as {na_equivalent_value = }")
    logger.info(f"{time_delta_column_name = }")

    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    # config end

    # Todo: Rudimentary Clean up (apacheversion iv vs iva) um speicherplatzu zu sparen
    df_table: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path,
        debug_mode=debug_mode,
        engine="pyarrow",
    )

    logger.info(f"{df_table.dtypes = }")

    df_table = data_cleaning.replace_na_equivalent_values(
        df_in=df_table, value_to_replace=na_equivalent_value, description=True
    )

    # table specific cleanup
    if table_name == "apachepatientresult":
        df_table = data_cleaning.remove_wrong_apache_version_rows(df_in=df_table)

    if table_name == "patient":
        # gender
        df_table["gender"] = df_table["gender"].replace("", pd.NA)
        df_table = df_table[df_table["gender"].str.lower() != "other"]
        median_gender = df_table["gender"].value_counts().index[0]
        df_table["gender"] = df_table["gender"].replace(pd.NA, median_gender)
        df_table["gender"] = df_table["gender"].replace("Unknown", median_gender)
        # we will this here since the number of "other"s is negligible

        # ethnicity
        df_table["ethnicity"] = df_table["ethnicity"].replace("", pd.NA)

        # age
        df_table["age"] = df_table["age"].replace("> 89", "90")
        df_table["age"] = df_table["age"].replace("", pd.NA)

        # df_age = df_table["age"].convert_dtypes()
        df_age = df_table["age"].astype("Int32")
        logger.info(f"Patient table age column stats {df_age.describe()}")

        # admission height
        df_table["admissionheight"] = df_table["admissionheight"].astype("Float32")

        mask_feet_lower = (df_table["admissionheight"] > 20) & (
                df_table["admissionheight"] < 70
        )
        mask_feet_upper = df_table["admissionheight"] > 250

        # we will assume that 20 < x < 70 would represent 2 < x < 7 feet, for lower feet conversion
        df_table.loc[mask_feet_lower, "admissionheight"] = (
                df_table.loc[mask_feet_lower, "admissionheight"] * 0.1 * 0.0254
        )

        # we will assume that 250 < x represent 2.5 < x feet  for upper feet conversion
        df_table.loc[mask_feet_upper, "admissionheight"] = (
                df_table.loc[mask_feet_upper, "admissionheight"] * 0.01 * 0.0254
        )

        # fix for people that are apparently 20 cm tall
        df_table["admissionheight"] = df_table["admissionheight"].mask(
            df_table["admissionheight"] < 20, pd.NA
        )

        # admission weight
        # we'll assume that its pounds above 400 instead of kg
        df_table["admissionweight"] = df_table["admissionweight"].astype("Float32")

        mask_pounds = df_table["admissionweight"] > 400

        df_table.loc[mask_pounds, "admissionweight"] = (
                df_table.loc[mask_pounds, "admissionweight"] * 0.453592
        )

        df_table["admissionweight"] = df_table["admissionweight"].mask(
            df_table["admissionweight"] < 20, pd.NA
        )

    if table_name == "vitalperiodic":
        # fahrenheit to celsius
        # https://en.wikipedia.org/wiki/Human_body_temperature

        # lower bound 13.7
        df_table["temperature"] = df_table["temperature"].mask(
            df_table["temperature"] < 14, pd.NA
        )

        mask_temp = df_table["temperature"] > 48

        df_table.loc[mask_temp, "temperature"] = (
                (df_table.loc[mask_temp, "temperature"] - 32) * 5.0 / 9.0
        )

        # higher bound 46.5 C
        df_table["temperature"] = df_table["temperature"].mask(
            df_table["temperature"] > 47, pd.NA
        )

        # st segments level, 0.1 is already associated with a major problem

        st_seg_threshold = 5

        df_table["st1"] = df_table["st1"].mask(
            (df_table["st1"] > st_seg_threshold) | (df_table["st1"] < -st_seg_threshold),
            pd.NA,
        )

        df_table["st2"] = df_table["st2"].mask(
            (df_table["st2"] > st_seg_threshold) | (df_table["st2"] < -st_seg_threshold),
            pd.NA,
        )

        df_table["st3"] = df_table["st3"].mask(
            (df_table["st3"] > st_seg_threshold) | (df_table["st3"] < -st_seg_threshold),
            pd.NA,
        )

        # blood and pulmonary artery pressure
        df_table["systemicsystolic"] = df_table["systemicsystolic"].mask(
            df_table["systemicsystolic"] <= 0, pd.NA
        )

        df_table["systemicdiastolic"] = df_table["systemicdiastolic"].mask(
            df_table["systemicdiastolic"] <= 0, pd.NA
        )

        df_table["systemicmean"] = df_table["systemicmean"].mask(
            df_table["systemicmean"] <= 0, pd.NA
        )

        df_table["pasystolic"] = df_table["pasystolic"].mask(
            df_table["pasystolic"] <= 0, pd.NA
        )

        df_table["padiastolic"] = df_table["padiastolic"].mask(
            df_table["padiastolic"] <= 0, pd.NA
        )

        df_table["pamean"] = df_table["pamean"].mask(df_table["pamean"] <= 0, pd.NA)

        # percentile outlier removal
        skip_columns = ["patientunitstayid", "observationoffset", "vitalperiodicid"]
        df_table = data_cleaning.set_outliers_to_na(
            df=df_table,
            lower_percentile=0.001,
            upper_percentile=0.999,
            skip_columns=skip_columns,
        )

    if table_name == "vitalaperiodic":
        df_table["svr"] = df_table["svr"].mask(df_table["svr"] <= 0, pd.NA)
        df_table["svri"] = df_table["svri"].mask(df_table["svri"] <= 0, pd.NA)
        df_table["pvr"] = df_table["pvr"].mask(df_table["pvr"] <= 0, pd.NA)
        df_table["pvri"] = df_table["pvri"].mask(df_table["pvri"] <= 0, pd.NA)

        skip_columns = ["patientunitstayid", "observationoffset", "vitalaperiodicid"]
        df_table = data_cleaning.set_outliers_to_na(
            df=df_table,
            lower_percentile=0.0001,
            upper_percentile=0.9999,
            skip_columns=skip_columns,
        )

    if time_delta_column_name != "":
        df_table = df_table.rename(columns={time_delta_column_name: "time_offset"})

        time_lower_bound_h: int = 0
        time_upper_bound_h: int = 168

        df_table["time_offset"] = pd.to_timedelta(df_table["time_offset"], unit="min")

        time_stats_pre = df_table["time_offset"].describe()

        logger.info(f"{time_stats_pre = }")

        df_table = df_table[
            (df_table["time_offset"] >= pd.to_timedelta(time_lower_bound_h, unit="h")) & (
                    df_table["time_offset"] <= pd.to_timedelta(time_upper_bound_h, unit="h"))]

        time_stats_post = df_table["time_offset"].describe()

        logger.info(f"{time_stats_post = }")

    logger.warning(f"Saved table stats: \n {df_table.describe() = }")

    data_handling.save_parquet_table(
        df_in=df_table, save_path=data_output_dir, filename=output_filename
    )

    logger.success("Table basic cleanup done !")


if __name__ == "__main__":
    main()
