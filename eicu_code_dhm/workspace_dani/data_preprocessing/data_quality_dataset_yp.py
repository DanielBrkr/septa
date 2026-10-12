"""This is using ydata profiling instead of AutoViz"""

import sys

import pandas as pd
from pathlib import Path
from loguru import logger
import hydra

from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf

from ydata_profiling import ProfileReport

# fixes import errors related to workspace_dani
sys.path.insert(1, str(Path.cwd().resolve()))

from workspace_dani.data_preprocessing.utils import data_handling


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
    draft_mode: bool = cfg.draft_mode
    data_quality_stage: str = cfg.data.which_stage
    yd_ts_mode: bool = cfg.data.yd_tsmode

    # overall data directory
    data_dir_path = Path(cfg.directories.data)

    # table specific configuration

    data_filename: str = cfg.data.dataset.table_name

    data_root: Path = Path(cfg.directories.outputs_root)

    data_outputs_root: str = cfg.directories.outputs_root
    percentiles: list = cfg.data.settings.percentiles

    logger.info(f"{percentiles = }")

    parquet_data_path: Path = data_root / Path(cfg.data.dataset.paths.table_location)

    data_output_dir: Path = Path(data_outputs_root) / Path(
        cfg.data.directories.data_save_dir
    )

    output_filename: str = f"{data_filename}"

    output_table_file_path: Path = data_output_dir / Path(output_filename)

    output_table_file_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Data directory: {Path(data_dir_path).resolve()}")
    logger.info(f"Loading file: {parquet_data_path}")

    logger.info(f"{data_outputs_root = }")
    logger.info(f"{data_output_dir = }")
    logger.info(f"{output_table_file_path = }")

    # save path setup for individual files
    report_output_html_path: Path = output_table_file_path / Path(
        data_filename + "data_profiling.html"
    )

    logger.info(f"{report_output_html_path = }")
    # config end

    df_table: pd.DataFrame = data_handling.load_parquet_table(
        parquet_path=parquet_data_path, debug_mode=debug_mode, backend="numpy_nullable"
    )

    # this is hack such that autoviz doesn't fail to an ambiguous boolean comparison due to the NA dtype
    # df: pd.DataFrame = df.replace(to_replace=[-1.0], value=np.nan)

    # the time delta column has been removed in this step, not yet useful
    # if time_delta_column_name != "":
    #        df = df.drop(time_delta_column_name, axis=1)

    logger.info(f"{'-' * 100}")

    if yd_ts_mode:
        data_report = ProfileReport(df_table, title=f"{data_filename}", tsmode=True)
    else:
        data_report = ProfileReport(df_table, title=f"{data_filename}", tsmode=False)

    # CSV and parquet saving
    # Note: Parquet saving seemed pretty buggy for this kind of data
    if not draft_mode:
        data_report.to_file(report_output_html_path)

    logger.success("Data Quality assessment done!")


if __name__ == "__main__":
    tqdm.pandas()
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG", diagnose=True)

    main()
