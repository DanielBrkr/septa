import glob
import io
import os.path

import panel as pn
import pandas as pd
import hvplot.pandas
from loguru import logger
from pathlib import Path

pn.extension("tabulator", design="material", template="material", loading_indicator=True)

csv_path = "../intermediary_data/apache_aps_var_wo_ids.csv"
INTER_DATA_PATH = Path("..") / Path("intermediary_data")

logger.debug(f"INTER_DATA_PATH: {INTER_DATA_PATH.resolve()}")

csv_pattern = os.path.join(INTER_DATA_PATH, "*.csv")
csv_files = glob.glob(csv_pattern)

plot_backends = ["bokeh", "matplotlib", "plotly"]
backend_selector = pn.widgets.Select(
    name="Plot Renderer", options=plot_backends, value=plot_backends[0]
)


@pn.depends(backend=backend_selector)
def set_plotting_backend(backend):
    hvplot.extension(backend)


file_selector = pn.widgets.Select(name="Choose CSV File", options=csv_files, value=csv_files[0])

column_selector = pn.widgets.CheckBoxGroup(
    name="Column Selector",
    inline=False,
)


@pn.depends(file_path=file_selector)
def get_data_csv(file_path=None):
    if not file_path:
        file_path = ""
        logger.debug(f"No file path specified, falling back to default")
        df = pd.read_csv(csv_path)
        new_columns = df.columns.to_list()
        logger.debug(f"New columns from default file : {new_columns}")
        column_selector.options = new_columns
        # column_selector.value = new_columns        # Note: Can't uncheck boxes when default assigned

    else:
        logger.debug(f"Filepaths from file selector: {file_path}")
        df = pd.read_csv(file_path)
        new_columns = df.columns.to_list()
        logger.debug(f"New columns from file selector : {new_columns}")
        column_selector.options = new_columns
        # column_selector.value = new_columns       # Note: Can't uncheck boxes when default assigned

    return df, new_columns


@pn.depends(columns_to_evaluate=column_selector, file_path=file_selector)
def explore(columns_to_evaluate, file_path):
    logger.debug(f"Explorer Fct: file_path={file_path}")

    df, init_columns = get_data_csv(file_path)

    logger.debug(f"Explorer Fct: dataframe head: \n {df.head(n=1)}")

    if len(columns_to_evaluate) < 1:
        selected_columns = init_columns
    else:
        selected_columns = columns_to_evaluate

    logger.debug(f"Explorer Fct: selected_columns={selected_columns}")

    df_to_evaluate = df[selected_columns]

    explorer = hvplot.explorer(df_to_evaluate)

    return pn.Column(explorer)


ACCENT = "#BB2649"

text = """Select the columns that should be plotted.
        Every column will be plotted by default. 
        Check the individual boxes according to your need.
        """
column_selection_box = pn.layout.WidgetBox(
    pn.pane.Markdown(text, sizing_mode="stretch_width"),
    column_selector,
)

file_backend_panel = pn.Row(
    pn.Column(file_selector), pn.Column(backend_selector), sizing_mode="stretch_both"
)

dataset_panel = pn.Row(column_selection_box, scroll=True)
explorer_panel = pn.Row(explore, sizing_mode="stretch_both")
author_panel = pn.Column(pn.pane.LaTeX("by Daniel Breucker"))

panel_page = pn.template.FastGridTemplate(
    sidebar=[author_panel, dataset_panel],
    sidebar_width=200,
    title="eICU CSV Data Viewer",
    accent_base_color=ACCENT,
    header_background=ACCENT,
    prevent_collision=True,
    save_layout=True,
    theme_toggle=False,
    row_height=160,
    meta_author="Daniel Breucker",
)

panel_page.main[0:1, 0:12] = file_backend_panel
panel_page.main[1:7, 0:12] = explorer_panel

panel_page.show()
