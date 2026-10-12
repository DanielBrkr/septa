import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
import pickle
import hydra
from adbc_driver_postgresql import dbapi

from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf

# fixes import errors related to workspace_dani
sys.path.insert(1, os.getcwd())

tqdm.pandas()


def convert_to_timedelta(stay, table_name: str, offset_column: str):
    time_delta_column_name: str = "time_delta_" + table_name
    stay[time_delta_column_name] = pd.to_timedelta(
        stay[offset_column] - stay[offset_column].iloc[0], unit="min"
    )

    return stay


@hydra.main(
    version_base=None, config_path="../conf", config_name="config_table_generation"
)
def main(cfg: DictConfig):
    logger.info(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    # inter_data_path = Path("..") / Path("..") / Path("intermediary_data")  # Todo: hydra
    inter_data_path: Path = Path(cfg.directories.data)
    inter_data_path_ids: Path = Path(cfg.directories.ids)
    inter_data_path_save_path: Path = Path(cfg.directories.save)

    logger.info(f"Loading from {inter_data_path.resolve()}")
    logger.info(f"Loading patientunitstayids from {inter_data_path_ids.resolve()}")
    logger.info(f"Saving to {inter_data_path_save_path.resolve()}")

    ids_path: Path = inter_data_path_ids / Path(cfg.files.id_file)
    with open(ids_path, "rb") as f:
        patient_stays_valid = pickle.load(f)

    logger.info(f"Sample of valid patient unit stay ids {patient_stays_valid[:8]}")

    table: str = cfg.table.file_name
    offset_name = cfg.table.time_offset_name

    conn_url_right: str = cfg.db_url
    row_limit = cfg.row_limit

    if row_limit is None:
        # "NULL" in yaml is being converted to NONE when using the hydra CLI
        # this is not intendend behaviour
        row_limit = "NULL"
    else:
        logger.warning(
            f"Using numbers of table rows is limited to {row_limit}, check if this is intended!"
        )

    data_overrides: dict = cfg.table.data_type_overrides
    data_type_backend = cfg.data_type_backend
    enable_data_type_overrides: bool = cfg.enable_data_type_overrides

    logger.info(f"Data type overrides: {data_overrides}")
    logger.info(f"{data_type_backend = }")
    logger.info(f"{enable_data_type_overrides = }")
    logger.info(f"Connecting to database via following URL: {conn_url_right}")

    if offset_name == "":
        logger.info("No time offset in table!")

        # noinspection SqlNoDataSourceInspection,SqlDialectInspection
        query_valid_apache_wo_schema = f"""  
        SELECT *
        FROM {"eicu_crd." + cfg.table.table_name}
--         WHERE patientunitstayid in ({", ".join(map(str, patient_stays_valid))}) 
        ORDER BY patientunitstayid
        LIMIT {row_limit}
        """

    else:
        # noinspection SqlNoDataSourceInspection,SqlDialectInspection
        query_valid_apache_wo_schema = f"""  
        SELECT *
        FROM {"eicu_crd." + cfg.table.table_name}
--         WHERE patientunitstayid in ({", ".join(map(str, patient_stays_valid))}) 
        ORDER BY patientunitstayid, {cfg.table.time_offset_name}
        LIMIT {row_limit}
        """
    # if data_type_backend == "pyarrow":
    #     data_overrides = {k: v + "[pyarrow]" for k, v in data_overrides.items()}

    if enable_data_type_overrides:
        logger.info(f"Actual data type overrides are: {data_overrides}")
        with dbapi.connect(conn_url_right) as conn:
            df_table = pd.read_sql(
                sql=query_valid_apache_wo_schema,
                con=conn,
                # dtype_backend=data_type_backend,
                dtype_backend=data_type_backend,
                dtype=data_overrides,
            )
    else:
        with dbapi.connect(conn_url_right) as conn:
            df_table = pd.read_sql(
                sql=query_valid_apache_wo_schema,
                con=conn,
                dtype_backend=data_type_backend,
            )

    logger.info(f"Head of the dataframe: \n{df_table.head()}")
    logger.info(f"Dataframe info: \n{df_table.info()}")

    # Todo: Add this in an later step, the dataframe foot print is much smaller this way
    # if offset_name == "":
    #     logger.info("Skipping addition of time_delta column!")
    # else:
    #     logger.info("Adding time delta column:")
    #     df_table = df_table.groupby("patientunitstayid").progress_apply(
    #         convert_to_timedelta, table_name=table, offset_column=offset_name).reset_index(drop=True)
    #
    # logger.info(f"The time delta column has been added to the dataframe."
    #             f"Head of the resulting dataframe: \n{df_table.head()}")
    # logger.info(f"Resulting dataframe info: \n{df_table.info()}")

    filename: str = "_".join([table, cfg.table.file_suffix]) + ".parquet"

    date = cfg.metadata.timestamp

    parquet_save_dir_path = inter_data_path_save_path / date
    parquet_save_dir_path.mkdir(parents=True, exist_ok=True)
    parquet_save_path = Path(inter_data_path_save_path / date / filename)

    df_table.to_parquet(path=parquet_save_path.resolve())

    logger.info(f"Saved dataframe to {parquet_save_path}")
    logger.success(f"Dataframe {table} has been saved as a parquet file.")


if __name__ == "__main__":
    main()
