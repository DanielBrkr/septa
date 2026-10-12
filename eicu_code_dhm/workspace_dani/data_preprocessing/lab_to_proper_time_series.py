import pandas as pd
import polars as pl
from pathlib import Path
from loguru import logger
import hydra
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

from utils import data_handling


# pycharm run config
# cohort@data.cohort=sepsis_type_b data=lab_proper_ts_cohort_b table@data.cohort.table=lab debug_mode=True draft_mode=False
#@logger.catch
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
    data_filename: str = cfg.data.cohort.table.file_name
    # data_filename: str = cfg.data.file_name
    table_name: str = cfg.data.cohort.table.table_name

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    data_input_dir: Path = data_root / Path(cfg.data.directories.input_data_dir)

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp

    parquet_data_path: Path = data_input_dir / Path(data_filename + ".parquet")
    data_output_dir: Path = Path(data_root) / Path(save_sub_dir) / Path(date)

    data_output_dir.mkdir(parents=True, exist_ok=True)
    # output_filename: str = "".join([data_filename])
    # output_filename: str = f"{data_filename}_consolidated"
    output_filename: str = f"{data_filename}"

    data_root: Path = Path(cfg.directories.outputs_root)

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading file: {parquet_data_path}")
    logger.info(f"{table_name = }")

    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    # config end

    df_lab: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path, debug_mode=debug_mode, backend="numpy_nullable"
    )#.head(n=10000) # whoa this f'd me up so hard

    df_lab_dummies = pd.get_dummies(
        df_lab[["patientunitstayid", "time_offset", "labname", "labresult"]]
    )

    df_dummies_transf = pd.DataFrame(columns=df_lab_dummies.columns)

    dummies_index_unique = df_lab_dummies.drop_duplicates(
        subset=["patientunitstayid", "time_offset"]
    )[["patientunitstayid", "time_offset"]]

    df_dummies_transf_ind = pd.concat(
        [dummies_index_unique, df_dummies_transf]
    ).reset_index(drop=True)

    df_dummies_transf_ind.set_index(["patientunitstayid", "time_offset"], inplace=True)

    if use_pandas := True:
        for index, row in tqdm(df_lab_dummies.iterrows(), total=len(df_lab_dummies)):
            id = row["patientunitstayid"]
            time_stamp = row["time_offset"]
            true_col = row[row == True].index.tolist()[0]
            measurement = row["labresult"]

            df_dummies_transf_ind.loc[(id, time_stamp), true_col] = measurement

    else:
        # Note: Dont use this, there was some breaking change, pandas is good enough
        df_lab_dummies_pl = pl.from_pandas(df_lab_dummies)
        df_dummies_transf_ind_pl = pl.from_pandas(
            df_dummies_transf_ind, include_index=True
        )

        for row in tqdm(
            df_lab_dummies_pl.iter_rows(named=True), total=df_lab_dummies_pl.height
        ):
            id = row["patientunitstayid"]
            time_stamp = row["time_offset"]
            true_col = [col for col, value in row.items() if value is True][0]
            measurement = row["labresult"]

            # Update the corresponding row in the Polars DataFrame
            mask = (df_dummies_transf_ind_pl["patientunitstayid"] == id) & (
                df_dummies_transf_ind_pl["time_offset"] == time_stamp
            )
            df_dummies_transf_ind_pl = df_dummies_transf_ind_pl.with_columns(
                pl.when(mask)
                .then(df_dummies_transf_ind_pl[true_col].set(measurement))
                .otherwise(df_dummies_transf_ind_pl[true_col])
                # .alias(true_col)
            )

            # print("hallo")

    # df_lab_proper_time_series = df_dummies_transf_ind.drop(columns="time_offset")
    df_lab_proper_time_series = df_dummies_transf_ind
    df_lab_proper_time_series = df_lab_proper_time_series.reset_index()
    df_lab_proper_time_series = df_lab_proper_time_series.drop(columns=["labresult"])

    data_handling.save_parquet_table(
        df_in=df_lab_proper_time_series,
        save_path=data_output_dir,
        filename=output_filename,
    )

    logger.warning(f"Lab to proper TS: saved table stats: \n {df_lab_proper_time_series.describe() = }")
    logger.success(f" {data_filename} time series data has been consolidated!")


if __name__ == "__main__":
    main()
