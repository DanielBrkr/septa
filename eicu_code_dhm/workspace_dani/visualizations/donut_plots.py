import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import palettable
from loguru import logger

from matplotlib.ticker import EngFormatter


def make_donut_plot(
    data_values: list,
    data_percentages: list | None,
    data_labels: list,
    value_fmt: str = "1.1%",
    use_value_fmt: bool = True,
    use_eng_fmt: bool = False,
    plot_title: str = "",
    hatch_flag: bool = True,
    show_plot: bool = True,
    x_text_modifier: float = 1.2,
    y_text_modifier: float = 1.3,
):
    with plt.rc_context({"hatch.color": "whitesmoke", "hatch.linewidth": 0.2}):
        fig_donut, ax_donut = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))

        color_palette = sns.color_palette("inferno", len(data_values))

        if len(data_values) > 20 and hatch_flag:
            logger.warning(
                f"The number of hatches exceeds the number of available hatch types with {len(data_values) = }, "
                f"you should really consider plotting without hatches."
            )

        if hatch_flag:
            hatches: list | None = [
                "/",
                "\\",
                "|",
                "-",
                "+",
                "x",
                "o",
                "O",
                ".",
                "*",
                "/o",
                "\\|",
                "|*",
                "-\\",
                "+o",
                "x*",
                "o-",
                "O|",
                "O.",
                "*-",
            ]
        else:
            hatches = None

        wedges, texts = ax_donut.pie(
            data_values,
            wedgeprops=dict(width=0.5),
            startangle=-40,
            colors=color_palette,
            hatch=hatches,
        )

        kw = dict()

        eng_fmt = EngFormatter(places=1, sep="\N{THIN SPACE}")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))

            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

            x_text_mod = x_text_modifier
            y_text_mod = y_text_modifier

            x_text = x_text_mod * float(np.sign(x))
            y_text = y_text_mod * y

            label = data_labels[i]
            value_raw = data_values[i]

            if len(data_percentages) != 0:
                value_perc = data_percentages[i]

            if not use_eng_fmt:
                value = f"{value_raw:{value_fmt}}"
            elif use_eng_fmt and use_value_fmt and len(data_percentages) != 0:
                value = f"{value_perc:{value_fmt}}" + f", {eng_fmt(value_raw)}"
            else:
                value = eng_fmt(value_raw)

            # var arrows
            ax_donut.annotate(
                f"{label} ({value})",
                xy=(x, y),
                xytext=(x_text, y_text),
                horizontalalignment=horizontalalignment,
                annotation_clip=False,
                size=15,
                arrowprops=dict(
                    arrowstyle="->",
                    edgecolor="k",
                    facecolor="black",
                    lw=2,
                    connectionstyle=f"angle,angleA=0,angleB={ang}",
                ),
                **kw,
            )
        plt.title(plot_title, size=18)
        plt.tight_layout()

        if show_plot:
            plt.show()


def make_nested_donut_plot(
    data_values: list,
    data_percentages: list | None,
    data_labels: list,
    value_fmt: str = "1.1%",
    use_value_fmt: bool = True,
    use_eng_fmt: bool = False,
    plot_title: str = "",
    hatch_flag: bool = True,
    show_plot: bool = True,
    x_text_modifier: float = 1.2,
    y_text_modifier: float = 1.3,
):
    with plt.rc_context({"hatch.color": "whitesmoke", "hatch.linewidth": 0.2}):
        fig_donut, ax_donut = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))

        color_palette = sns.color_palette("inferno", len(data_values))

        if len(data_values) > 20 and hatch_flag:
            logger.warning(
                f"The number of hatches exceeds the number of available hatch types with {len(data_values) = }, "
                f"you should really consider plotting without hatches."
            )

        if hatch_flag:
            hatches: list | None = [
                "/",
                "\\",
                "|",
                "-",
                "+",
                "x",
                "o",
                "O",
                ".",
                "*",
                "/o",
                "\\|",
                "|*",
                "-\\",
                "+o",
                "x*",
                "o-",
                "O|",
                "O.",
                "*-",
            ]
        else:
            hatches = None

        wedges, texts = ax_donut.pie(
            data_values,
            wedgeprops=dict(width=0.5),
            startangle=-40,
            colors=color_palette,
            hatch=hatches,
        )

        kw = dict()

        eng_fmt = EngFormatter(places=1, sep="\N{THIN SPACE}")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))

            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

            x_text_mod = x_text_modifier
            y_text_mod = y_text_modifier

            x_text = x_text_mod * float(np.sign(x))
            y_text = y_text_mod * y

            label = data_labels[i]
            value_raw = data_values[i]

            if len(data_percentages) != 0:
                value_perc = data_percentages[i]

            if not use_eng_fmt:
                value = f"{value_raw:{value_fmt}}"
            elif use_eng_fmt and use_value_fmt and len(data_percentages) != 0:
                value = f"{value_perc:{value_fmt}}" + f", {eng_fmt(value_raw)}"
            else:
                value = eng_fmt(value_raw)

            # var arrows
            ax_donut.annotate(
                f"{label} ({value})",
                xy=(x, y),
                xytext=(x_text, y_text),
                horizontalalignment=horizontalalignment,
                annotation_clip=False,
                size=15,
                arrowprops=dict(
                    arrowstyle="->",
                    edgecolor="k",
                    facecolor="black",
                    lw=2,
                    connectionstyle=f"angle,angleA=0,angleB={ang}",
                ),
                **kw,
            )
        plt.title(plot_title, size=18)
        plt.tight_layout()

        if show_plot:
            plt.show()
