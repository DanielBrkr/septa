import holoviews as hv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import colors
import pandas as pd
from holoviews import opts
import numpy as np
from loguru import logger
import seaborn as sns

# from dani_plotting import daniplot
import timple
from tqdm.auto import tqdm


from workspace_dani.data_preprocessing.utils.data_quality import (
    generator_parameter_coverage_ratio,
)


def cm2inch(value):
    """converts cm to inch for plotting
    :param float value:
    """
    return value / 2.54


def figure_setup(
    num_row=1, num_cols=1, width_cm=16, height_cm=5
) -> (plt.Figure, plt.Axes, plt.rc):
    # plot setup
    plt.rcParams["font.family"] = "Helvetica"
    sns.set(style="ticks")

    fig, axes = plt.subplots(
        nrows=num_row,
        ncols=num_cols,
        dpi=300,
        figsize=(cm2inch(width_cm), cm2inch(height_cm)),
        squeeze=True,
    )

    if isinstance(axes, plt.Axes):
        # color_palette = palettable.cartocolors.qualitative.Pastel_6.mpl_colors
        # axes.set_prop_cycle(
        #     "color", palettable.cartocolors.qualitative.Pastel_6.mpl_colors
        # )
        sns.despine(ax=axes)
        axes.set_axisbelow(True)

    else:
        for axs in axes.flatten():
            # axs.set_prop_cycle(
            #     "color", palettable.cartocolors.qualitative.Pastel_6.mpl_colors
            # )
            sns.despine(ax=axs)
            axs.set_axisbelow(True)

    plt.xticks(rotation=30, ha="right")

    return fig, axes


def layout_hv_curves(
    dataframe_to_plot: pd.DataFrame, columns_to_plot: list[str], title: str
) -> hv.Layout:
    columns_with_time = ["observationoffset", *columns_to_plot]

    long_df = dataframe_to_plot[columns_with_time].melt(
        id_vars="observationoffset", var_name="Variable", value_name="Value"
    )

    hv_plot = (
        hv.Dataset(long_df)
        .to(hv.Curve, "observationoffset", "Value", groupby="Variable")
        .overlay()
    )

    xmin = dataframe_to_plot["observationoffset"].min()
    xmax = dataframe_to_plot["observationoffset"].max()

    vline_min = hv.VLine(xmin).opts(color="red", line_dash="dashed")
    vline_max = hv.VLine(xmax).opts(color="red", line_dash="dashed")

    hv_plot.opts(opts.Curve(width=720, tools=["hover"]))
    hv_plot.opts(autorange="x", xlim=(xmin, xmax))
    hv_plot.opts(title=title)
    hv_plot.opts({"Curve": {"color": hv.Palette("Plasma")}})

    grid_style = {
        "grid_line_color": "lightgray"  # , "minor_xgrid_line_color": 'lightgray',
    }
    hv_plot.opts(gridstyle=grid_style, show_grid=True)

    layout_with_bounds = hv_plot * vline_min * vline_max

    return layout_with_bounds


def layout_hv_scatter(
    dataframe_to_plot: pd.DataFrame, columns_to_plot: list[str], title: str
) -> hv.Layout:
    columns_with_time = ["observationoffset", *columns_to_plot]

    long_df = dataframe_to_plot[columns_with_time].melt(
        id_vars="observationoffset", var_name="Variable", value_name="Value"
    )

    hv_plot = (
        hv.Dataset(long_df)
        .to(hv.Scatter, "observationoffset", "Value", groupby="Variable")
        .overlay()
    )

    xmin = dataframe_to_plot["observationoffset"].min()
    xmax = dataframe_to_plot["observationoffset"].max()

    vline_min = hv.VLine(xmin).opts(color="red", line_dash="dashed")
    vline_max = hv.VLine(xmax).opts(color="red", line_dash="dashed")

    hv_plot.opts(opts.Curve(width=720, tools=["hover"]))
    hv_plot.opts(autorange="x", xlim=(xmin, xmax))
    hv_plot.opts(title=title)
    hv_plot.opts({"Curve": {"color": hv.Palette("Plasma")}})

    grid_style = {
        "grid_line_color": "lightgray"  # , "minor_xgrid_line_color": 'lightgray',
    }
    hv_plot.opts(gridstyle=grid_style, show_grid=True)

    layout_with_bounds = hv_plot * vline_min * vline_max

    return layout_with_bounds


def plot_matplotlib_curves(
    dataframe_to_plot: pd.DataFrame, columns_to_plot: list[str]
) -> hv.Layout:
    hv.extension("matplotlib")

    columns_with_time = ["observationoffset", *columns_to_plot]

    long_df = dataframe_to_plot[columns_with_time].melt(
        id_vars="observationoffset", var_name="Variable", value_name="Value"
    )

    hv_plot = (
        hv.Dataset(long_df)
        .to(hv.Curve, "observationoffset", "Value", groupby="Variable")
        .overlay()
    )

    xmin = dataframe_to_plot["observationoffset"].min()
    xmax = dataframe_to_plot["observationoffset"].max()

    vline_min = hv.VLine(xmin).opts(color="red")
    vline_max = hv.VLine(xmax).opts(color="red")

    hv_plot.opts(opts.Curve(title="Vital 1 Periodic TS"))
    hv_plot.opts({"Curve": {"color": hv.Palette("Plasma")}})

    layout_with_bounds = hv_plot * vline_min * vline_max

    return layout_with_bounds


def five_min_to_hours(x, pos):
    total_seconds = 5 * 60 * int(x)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    # return f"{hours}:{minutes:02d}"
    return f"{hours}"


def min_to_hours(x, pos):
    total_seconds = 60 * int(x)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    # return f"{hours}:{minutes:02d}"
    return f"{hours}"


def plot_vitalperiodic_ts_heatmap(
    frame_with_time_delta: pd.DataFrame,
    variable_col,
    time_delta_col: str,
    num_stays: int = 50,
    limit_hours: int = None,
    meas_unit: str = "",
    plot_holo: bool = False,
    fig_width_cm: int = 10,
    fig_height_cm: int = 10,
    **kwargs,
) -> (plt.Figure, hv.QuadMesh):
    """

    Args:
        plot_holo: plots also an interactive plot
        fig_height: mpl height
        fig_width: mpl width
        frame_with_time_delta: Has to contain the "time_delta" column
        variable_col: Single TS variable

    Returns: A plot containing the TS heatmap

    """

    # df_list = []
    # for stay in frame["patientunitstayid"].unique():
    #    df_temp = frame.loc[frame["patientunitstayid"] == stay]
    #    df_temp_var_time = df_temp[[variable_col, "time_delta"]]
    #    df_temp_var_time = df_temp_var_time.set_index("time_delta")
    #    df_list.append(df_temp_var_time)

    def process_patientunitstayid_group(group: pd.DataFrame, var_col: str, time_col: str):
        # group = group.set_index("time_delta")
        group = group.set_index(time_col)

        if group.index.hasnans:
            logger.warning("NaN in index")

        return group[var_col]

    investigated_parameter_df_list = [
        process_patientunitstayid_group(
            group, var_col=variable_col, time_col=time_delta_col
        )
        for _, group in frame_with_time_delta.groupby("patientunitstayid")
    ]

    # sns.set(style="whitegrid", font="TUM Neue Helvetica")

    df_var_time_conc: pd.DataFrame = pd.concat(
        investigated_parameter_df_list[0:num_stays], axis=1, ignore_index=True
    )

    if limit_hours != None:
        delta_timesteps: int = int((limit_hours * 60) / 5)
    else:
        delta_timesteps: int = len(df_var_time_conc)

    num_patient_stays: int = len(df_var_time_conc.columns)

    logger.info(f"Number of time delta steps: {delta_timesteps}")
    logger.info(f"Number of patient stays: {num_patient_stays}")

    if df_var_time_conc.isna().values.any():
        # pcolormesh does not work with pandas.NA but with regular NaN
        # this is problem when working with pandas nullable Int/Float types
        # df_var_time_conc: pd.DataFrame = df_var_time_conc.replace({pd.NA: np.nan})
        df_var_time_conc: pd.DataFrame = df_var_time_conc.astype("float")

    z = df_var_time_conc[0:delta_timesteps].T.to_numpy()
    z = z[:-1, :-1]

    xi, yi = np.meshgrid(
        np.linspace(0, delta_timesteps, delta_timesteps),
        np.linspace(0, num_patient_stays, num_patient_stays),
    )

    tmpl = timple.Timple()
    tmpl.enable()

    fig, axs, font_ticks, font_labels, font_titles = daniplot.plot_config_single(
        num_row=1, num_cols=1, height_cm=fig_height_cm, width_cm=fig_width_cm
    )

    # plot setup
    sns.set(style="whitegrid", font="TUM Neue Helvetica")

    mean_variable = frame_with_time_delta[variable_col].mean()
    normalization = colors.CenteredNorm(vcenter=mean_variable)
    c = axs.pcolormesh(xi, yi, z, norm=normalization, cmap="icefire")

    fig.colorbar(c, ax=axs)
    cbar = axs.collections[0].colorbar

    tick_font_size = 10
    label_font_size = 10
    title_font_size = 12

    # ticks
    five_min_hour_formatter = ticker.FuncFormatter(func=five_min_to_hours)
    integer_formater = ticker.FuncFormatter(lambda x, _: f"{int(x)}")
    hour_locator = mdates.HourLocator(interval=4)

    # locator = tmpldelta.AutoTimedeltaLocator()
    # formatter = tmpldelta.TimedeltaFormatter("%H:%m")
    # axs.xaxis.set_major_locator(locator)
    # axs.xaxis.set_major_formatter(formatter)

    # axs.xaxis.set_major_formatter(five_min_hour_formatter)
    # axs.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    # axs.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # axs.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Every hour
    # axs.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Only show hour and m

    # Todo: Gridlines nach hinten verschieben
    axs.xaxis.set_major_locator(plt.MultipleLocator(1 * 60))
    axs.xaxis.set_major_formatter(five_min_hour_formatter)
    axs.yaxis.set_major_formatter(integer_formater)

    axs.tick_params(axis="both", which="major", labelsize=tick_font_size)
    # axs.tick_params(axis='both', which='major', labelfontfamily=font_ticks)

    cbar.ax.tick_params(labelsize=tick_font_size)

    # labels
    plt.ylabel("Patient Unit Stay", fontsize=label_font_size)
    plt.xlabel("Time [h]", fontsize=label_font_size)

    cbar.set_label(f"{variable_col} [{meas_unit}]", fontsize=label_font_size)

    # titles
    # axs.set_title(f"{variable_col} across Patient Unit Stays", fontsize=title_font_size)

    # spines
    # axs.spines['top'].set_visible(False)
    # axs.spines['right'].set_visible(False)

    plt.tight_layout()

    # plt.close()

    if plot_holo:
        hv_heatmesh = hv.QuadMesh((xi, yi, z))

        hv_heatmesh.opts(
            cmap="inferno",
            colorbar=True,
            show_grid=True,
            width=1000,
            height=800,
            tools=["hover"],
            **kwargs,
        )
    else:
        hv_heatmesh = None

    return fig  # , hv_heatmesh


def plot_vitalperiodic_ts_heatmap_barebones(
    frame_with_time_delta: pd.DataFrame,
    variable_col: str,
    time_delta_col: str,
    num_stays: int = None,
    limit_hours: int = None,
    fig_width_cm: int = 10,
    fig_height_cm: int = 10,
    **kwargs,
) -> plt.Figure:
    def process_patientunitstayid_group(group: pd.DataFrame, var_col: str, time_col: str):
        # group = group.set_index("time_delta")
        group = group.set_index(time_col)

        if group.index.hasnans:
            logger.warning("NaN in index")

        return group[var_col]

    investigated_parameter_df_list = [
        process_patientunitstayid_group(
            group, var_col=variable_col, time_col=time_delta_col
        )
        for _, group in frame_with_time_delta.groupby("patientunitstayid")
    ]

    # sns.set(style="whitegrid", font="TUM Neue Helvetica")

    df_var_time_conc: pd.DataFrame = pd.concat(
        investigated_parameter_df_list[0:num_stays], axis=1, ignore_index=True
    )

    if limit_hours is not None:
        delta_timesteps: int = int((limit_hours * 60) / 5)
    else:
        delta_timesteps: int = len(df_var_time_conc)

    num_patient_stays: int = len(df_var_time_conc.columns)

    logger.info(f"Number of time delta steps: {delta_timesteps}")
    logger.info(f"Number of patient stays: {num_patient_stays}")

    if df_var_time_conc.isna().values.any():
        # pcolormesh does not work with pandas.NA but with regular NaN
        # this is problem when working with pandas nullable Int/Float types
        # df_var_time_conc: pd.DataFrame = df_var_time_conc.replace({pd.NA: np.nan})
        df_var_time_conc: pd.DataFrame = df_var_time_conc.astype("float")

    z = df_var_time_conc[0:delta_timesteps].T.to_numpy()
    z = z[:-1, :-1]

    xi, yi = np.meshgrid(
        np.linspace(0, delta_timesteps, delta_timesteps),
        np.linspace(0, num_patient_stays, num_patient_stays),
    )

    tmpl = timple.Timple()
    tmpl.enable()

    fig, axs = daniplot.plot_config_single(
        num_row=1, num_cols=1, height_cm=fig_height_cm, width_cm=fig_width_cm
    )

    mean_variable = frame_with_time_delta[variable_col].mean()
    normalization = colors.CenteredNorm(vcenter=mean_variable)
    c = axs.pcolormesh(xi, yi, z, norm=normalization, cmap="icefire")

    fig.colorbar(c, ax=axs)

    return fig


def plot_completeness_ratio_heatmap(
    df_data: pd.DataFrame,
    fig_width_cm: int,
    fig_height_cm: int,
):
    df_data_grouped: pd.DataFrame.groupby = df_data.groupby("patientunitstayid")
    df_data_grouped.head()

    columns_to_ignore = [
        "apacheapsvarid",
        "apachepredvarid",
        "apachepatientresultsid",
        "vitalperiodicid",
        "vitalaperiodicid",
        "patientunitstayid",
        "observationoffset",
        "labid",
        "labresultoffset",
        "labresultrevisedoffset",
        "intakeoutputid",
        "intakeoutputoffset",
        "intakeoutputentryoffset",
        "pasthistoryid",
        "pasthistoryoffset",
        "pasthistoryenteredoffset",
        "time_delta_past_history",
        "patienthealthsystemstayid",
        "diagnosisid",
        "diagnosisoffset",
        "",
    ]

    vp_parameters: list = []
    for column in df_data.columns:
        if column in columns_to_ignore:
            pass
        else:
            vp_parameters.append(column)

    parameter_coverage_ratio_list_all: list = []

    for coverage_ratio in tqdm(
        generator_parameter_coverage_ratio(
            df_grouped=df_data_grouped, parameters=vp_parameters
        ),
        total=len(df_data_grouped),
    ):
        parameter_coverage_ratio_list_all.append(coverage_ratio)

    parameter_coverage_ratio_df: pd.DataFrame = pd.DataFrame(
        data=parameter_coverage_ratio_list_all
    )

    fig, axs = daniplot.plot_config_single(
        num_row=1, num_cols=1, height_cm=fig_height_cm, width_cm=fig_width_cm
    )

    heatmap = sns.heatmap(
        parameter_coverage_ratio_df.drop(["patientunitstayid"], axis=1),
        ax=axs,
        # annot=False, cmap="rocket", xticklabels=True, yticklabels=True)
        annot=False,
        cmap="rocket",
        xticklabels=True,
        yticklabels="auto",
    )

    return fig, axs
