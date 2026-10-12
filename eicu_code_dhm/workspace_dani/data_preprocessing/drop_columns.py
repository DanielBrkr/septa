import sys
import pandas as pd
from pathlib import Path
from loguru import logger
import hydra

from omegaconf import DictConfig, OmegaConf

# fixes import errors related to workspace_dani
# sys.path.insert(1, str(Path.cwd()))

from utils import data_handling, data_cleaning


@logger.catch
@hydra.main(version_base=None, config_path="../conf", config_name="config_data.yaml")
def main(cfg: DictConfig):
    # logger.remove()
    # logger.add(sys.stdout, colorize=True)
    #  general config portion
    hydra_log_path: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/run.log"
    logger.add(f"{hydra_log_path}")

    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    debug_mode: bool = cfg.debug_mode

    # table specific configuration
    data_filename: str = cfg.data.table.file_name
    table_name: str = cfg.data.table.table_name

    # data processing specific configuration
    data_root: Path = Path(cfg.directories.outputs_root)
    data_input_dir: Path = data_root / Path(cfg.data.directories.input_data_dir)

    save_sub_dir = cfg.data.directories.data_save_dir

    date = cfg.metadata.timestamp

    basis_table_path: Path = Path(cfg.data.table.paths.table_location)
    parquet_data_path: Path = data_input_dir / basis_table_path.name
    data_output_dir: Path = Path(data_root) / Path(save_sub_dir) / Path(date)

    data_output_dir.mkdir(parents=True, exist_ok=True)
    output_filename: str = "".join([data_filename])

    dropping_columns: bool = cfg.data.table.dropping_columns
    columns_to_drop: list = cfg.data.table.columns_to_drop
    columns_to_keep: list = cfg.data.table.columns_to_keep

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Loading file: {parquet_data_path}")
    logger.info(f"{table_name = }")

    logger.info(f"{data_root = }")
    logger.info(f"{data_output_dir = }")

    # config end

    df_table: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path, debug_mode=debug_mode
    )

    df_table = data_cleaning.drop_unnecessary_columns(
        df_in=df_table,
        dropping_flag=dropping_columns,
        columns_drop=columns_to_drop,
        columns_keep=columns_to_keep,
    )

    data_handling.save_parquet_table(
        df_in=df_table, save_path=data_output_dir, filename=output_filename
    )

    logger.success("Dropping unnecessary columns done !")


if __name__ == "__main__":
    main()
