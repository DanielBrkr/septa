import os
import sys
from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa+
import hydra
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import timple
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pandarallel import pandarallel
from teeplot import teeplot as tp
from tqdm.auto import tqdm

# fixes import errors related to workspace_dani
sys.path.insert(1, os.getcwd())

from workspace_dani.visualizations.plot_helpers import (
    plot_vitalperiodic_ts_heatmap_barebones,
    five_min_to_hours,
)  # noqa+

# setup of packages
tqdm.pandas()
hv.extension("bokeh")


def convert_to_timedelta(stay):
    import pandas as pd

    stay["time_delta"] = pd.to_timedelta(
        stay["observationoffset"] - stay["observationoffset"].iloc[0], unit="min"
    )

    return stay


@hydra.main(version_base=None, config_path="../conf", config_name="config_plotting")
def plot_vital_periodic_ts_heatmap(cfg: DictConfig):
    #  general config portion
    logger.info(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")
    debug_mode: bool = cfg.debug_mode
    draft_mode: bool = cfg.draft_mode

    # overall data directory
    data_dir_path = Path(cfg.directories.data)

    # table specific configuration
    data_filename = cfg.plot.table.file_name
    table_variable_units: dict = cfg.plot.table.variable_units

    # plot specific configuration
    plot_type: str = cfg.plot.name
    plot_outputs_root: str = cfg.directories.plot_outputs_root

    column_of_interest = cfg.plot.settings.target
    time_delta_column: str = cfg.plot.table.time_delta_column
    got_timedelta: bool = cfg.plot.settings.got_timedelta

    plot_xaxis_label: str = cfg.plot.settings.xaxis_label
    plot_yaxis_label: str = cfg.plot.settings.yaxis_label
    plot_zaxis_label: str = table_variable_units[column_of_interest]
    plot_title: str = cfg.plot.settings.title

    number_stays: int = cfg.plot.settings.number_stays
    maximum_hours: int = cfg.plot.settings.limit_hours

    tum_font_scaling: float = cfg.plot.settings.tum_font_scaling

    parquet_data_path: Path = Path(cfg.plot.table.paths.table_location)
    plot_output_dir: Path = Path(plot_outputs_root) / Path(plot_type)

    show_plot: bool = cfg.plot.settings.show_plot

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Data directory: {Path(data_dir_path).resolve()}")
    logger.info(f"Loading file: {parquet_data_path}")

    logger.info(f"{plot_output_dir = }")
    logger.info(f"{plot_outputs_root = }")

    logger.info(f"Plotting Setup {'-' * 100}")
    logger.info(
        f"{plot_type = } | {column_of_interest = } | {time_delta_column = } | {got_timedelta = }"
    )
    logger.info(
        f"{number_stays = } | {maximum_hours = } | {plot_xaxis_label = } | {plot_yaxis_label = }"
    )
    logger.info(f"{plot_zaxis_label = } | {plot_title = }")

    if parquet_data_path.is_file():
        if debug_mode:
            df_vp_apache_valid = pd.read_parquet(
                parquet_data_path,
                engine="pyarrow",
                columns=[
                    "patientunitstayid",
                    "observationoffset",
                    column_of_interest,
                    time_delta_column,
                ],
            ).head(n=10000)
        else:
            df_vp_apache_valid = pd.read_parquet(
                parquet_data_path,
                engine="pyarrow",
                columns=[
                    "patientunitstayid",
                    "observationoffset",
                    column_of_interest,
                    time_delta_column,
                ],
            )

        logger.info(f"Loading data from local file")
        logger.info(f"Summary of the dataframe: \n {df_vp_apache_valid}")
        logger.info(f"Data types of the dataframe: \n {df_vp_apache_valid.info()}")

    else:
        logger.error(f"Dataframe vitalperiodic with valid apache not found")
        df_vp_apache_valid = None

    if not got_timedelta:
        df_vp_apache_valid = (
            df_vp_apache_valid.groupby("patientunitstayid")
            .progress_apply(
                # parallel doesn't do anything actually since we got only one column anyway
                # df_vp_apache_valid.groupby("patientunitstayid").parallel_apply(
                convert_to_timedelta
            )
            .reset_index(drop=True)
        )
        logger.info(f"Timedelta conversion done!")

    plot_output_dir.mkdir(parents=True, exist_ok=True)

    tmpl = timple.Timple()
    tmpl.enable()

    tp.draftmode = draft_mode

    saveit, figure = tp.tee(
        plotter=plot_vitalperiodic_ts_heatmap_barebones,
        frame_with_time_delta=df_vp_apache_valid,
        variable_col=column_of_interest,
        time_delta_col=time_delta_column,
        num_stays=number_stays,
        limit_hours=maximum_hours,
        fig_width_cm=10,
        fig_height_cm=10,
        plot_holo=False,
        teeplot_show=show_plot,
        teeplot_callback=True,
        teeplot_outinclude=["num_stays", "limit_hours"],
        teeplot_outdir=str(plot_output_dir.resolve()),
        teeplot_save={".eps", ".png", ".pdf"},
        teeplot_transparent=False,
    )

    axs = figure.axes[0]
    cbar = axs.collections[0].colorbar

    # fonts
    tick_font_size = 10 * tum_font_scaling
    label_font_size = 10 * tum_font_scaling
    title_font_size = 12 * tum_font_scaling

    # ticks
    five_min_hour_formatter = ticker.FuncFormatter(func=five_min_to_hours)
    integer_formater = ticker.FuncFormatter(lambda x, _: f"{int(x)}")

    axs.xaxis.set_major_locator(plt.MultipleLocator(1 * 60))
    axs.xaxis.set_major_formatter(five_min_hour_formatter)
    axs.yaxis.set_major_formatter(integer_formater)

    current_x_ticks = axs.get_xticks()
    current_x_ticks = [
        x for x in current_x_ticks if 0 <= x <= maximum_hours * 60 / 5
    ]  # drop negative tick and overextending last one

    current_y_ticks = axs.get_yticks()

    additional_x_tick = [(maximum_hours * 60) / 5]
    additional_y_tick = [number_stays]

    updated_x_ticks = np.sort(np.concatenate((current_x_ticks, additional_x_tick)))
    updated_y_ticks = np.sort(np.concatenate((current_y_ticks, additional_y_tick)))

    axs.set_xticks(updated_x_ticks)
    axs.set_yticks(updated_y_ticks)

    axs.tick_params(axis="both", which="major", labelsize=tick_font_size)

    cbar.ax.tick_params(labelsize=tick_font_size)

    # labels
    plt.ylabel("Patient Unit Stay", fontsize=label_font_size)
    plt.xlabel("Time [h]", fontsize=label_font_size)

    variable_col = column_of_interest
    time_delta_col = time_delta_column
    meas_unit = plot_zaxis_label

    cbar.set_label(f"{variable_col} [{meas_unit}]", fontsize=label_font_size)

    # titles
    # axs.set_title(f"{variable_col}", fontsize=title_font_size)

    # spines
    # axs.spines['top'].set_visible(False)
    # axs.spines['right'].set_visible(False)
    sns.despine()

    # grid lines
    plt.grid(True, which="both", axis="x")
    axs.set_axisbelow(True)

    plt.tight_layout()

    saveit()


if __name__ == "__main__":
    plot_vital_periodic_ts_heatmap()
