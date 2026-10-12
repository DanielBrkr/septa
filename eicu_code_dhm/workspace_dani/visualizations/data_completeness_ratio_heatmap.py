import sys
import os
from pathlib import Path


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf
from teeplot import teeplot as tp
from loguru import logger
from matplotlib.ticker import EngFormatter

# fixes import errors related to workspace_dani
sys.path.insert(1, os.getcwd())

from workspace_dani.visualizations.plot_helpers import plot_completeness_ratio_heatmap


# make sure to add "plot=completeness_ratio_heatmap" to the run config
@hydra.main(version_base=None, config_path="../conf", config_name="config_plotting")
def plot_completeness_heatmap(cfg: DictConfig):
    #  general config portion
    logger.info(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")
    debug_mode: bool = cfg.debug_mode
    draft_mode: bool = cfg.draft_mode

    # overall data directory
    data_dir_path = Path(cfg.directories.data)

    # table specific configuration
    data_filename = cfg.plot.table.file_name
    table_name_pretty = cfg.plot.table.table_name_pretty

    table_variables_pretty: dict = cfg.plot.table.variables_pretty

    # plot specific configuration
    plot_type: str = cfg.plot.name
    plot_outputs_root: str = cfg.directories.outputs_root

    time_offset_column: str = cfg.plot.table.time_offset_name
    time_delta_column: str = cfg.plot.table.time_delta_column

    plot_xaxis_label: str = cfg.plot.settings.xaxis_label
    plot_yaxis_label: str = cfg.plot.settings.yaxis_label
    plot_zaxis_label: str = cfg.plot.settings.zaxis_label
    plot_title: str = cfg.plot.settings.title

    tum_font_scaling: float = cfg.plot.settings.tum_font_scaling

    tick_font_size = cfg.plot.settings.tick_font_scale * tum_font_scaling
    label_font_size = cfg.plot.settings.label_font_scale * tum_font_scaling
    title_font_size = cfg.plot.settings.title_font_scale * tum_font_scaling

    width_cm = cfg.plot.settings.figure_width_cm
    height_cm = cfg.plot.settings.figure_height_cm
    resolution_dpi = cfg.plot.settings.resolution_dpi

    parquet_data_path: Path = Path(cfg.plot.table.paths.table_location)
    plot_output_dir: Path = Path(plot_outputs_root) / Path(plot_type)

    show_plot: bool = cfg.plot.settings.show_plot

    logger.info(f"Paths and Directories {'-' * 100}")
    logger.info(f"Data directory: {Path(data_dir_path).resolve()}")
    logger.info(f"Loading file: {parquet_data_path}")

    logger.info(f"{plot_output_dir = }")
    logger.info(f"{plot_outputs_root = }")

    logger.info(f"Plotting Setup {'-' * 100}")
    logger.info(f"{plot_type = } | {plot_xaxis_label = } | {plot_yaxis_label = }")
    logger.info(f"{plot_zaxis_label = } | {plot_title = }")

    if parquet_data_path.is_file():
        if debug_mode:
            df = pd.read_parquet(
                parquet_data_path,
                engine="pyarrow",
            ).head(n=10000)
        else:
            df = pd.read_parquet(
                parquet_data_path,
                engine="pyarrow",
            )

        logger.info("Loading data from local file")
        logger.info(f"Summary of the dataframe: \n {df}")
        logger.info(f"Data types of the dataframe: \n {df.info()}")

    else:
        logger.error(f"Parquet File not found! \n Assumed path: {parquet_data_path}")
        df = None

    if time_delta_column != "":
        df = df.drop(columns=time_delta_column)
        df = df.drop(columns=time_offset_column)

    else:
        pass

    tp.draftmode = draft_mode

    saveit, figure = tp.tee(
        plotter=plot_completeness_ratio_heatmap,
        df_data=df,
        fig_width_cm=width_cm,
        fig_height_cm=height_cm,
        teeplot_show=show_plot,
        teeplot_callback=True,
        # teeplot_outinclude=["table_name_pretty", str(debug_mode)],
        teeplot_outattrs={"table": table_name_pretty, "debug_mode": debug_mode},
        teeplot_outdir=str(plot_output_dir.resolve()),
        teeplot_save={".png", ".pdf"},
        teeplot_transparent=False,
        teeplot_dpi=resolution_dpi,
    )
    fig, axs = figure
    cbar = axs.collections[0].colorbar

    # ticks
    if debug_mode:
        pass
    else:
        axs.yaxis.set_major_locator(plt.MultipleLocator(1 * 10000))
        formatter1 = EngFormatter()
        axs.yaxis.set_major_formatter(formatter1)

    axs.tick_params(axis="both", which="major", labelsize=tick_font_size)
    plt.xticks(rotation=45, ha="right")

    axs.invert_yaxis()

    # labels
    pretty_x_labels = [
        table_variables_pretty.get(label.get_text()) for label in axs.get_xticklabels()
    ]

    plt.xlabel(table_name_pretty + " Variables", fontsize=label_font_size)
    axs.xaxis.set_label_position("top")

    plt.ylabel(plot_yaxis_label, fontsize=label_font_size)

    axs.set_xticklabels(pretty_x_labels)
    cbar.set_label(plot_zaxis_label, fontsize=label_font_size)

    # titles
    # axs.set_title(f"{variable_col}", fontsize=title_font_size)

    # spines
    sns.despine(ax=axs)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.5)

    # grid lines
    plt.grid(True, which="both", axis="x")
    axs.set_axisbelow(True)

    plt.tight_layout()

    saveit()

    logger.success("Plotting done!")


if __name__ == "__main__":
    plot_completeness_heatmap()
