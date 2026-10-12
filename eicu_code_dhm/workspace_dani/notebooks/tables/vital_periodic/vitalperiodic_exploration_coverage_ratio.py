from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa+

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

from workspace_dani.eicu_db_connection import DatabaseConnection


hv.extension("bokeh")

entire_dataset = True  # plot save flag and csv file decision
notebook_name = "vitalperiodic_exploration"

plot_outputs_dir_small = "subset"
plot_outputs_dir_all = "all"

INTER_DATA_PATH = Path("..") / Path("..") / Path("intermediary_data")

print(f"Loading and saving data from/to {Path(INTER_DATA_PATH).resolve()}")
connection = DatabaseConnection()
engine = connection.engine
query_schema = connection.query_schema


conn = engine.connect().execution_options(stream_results=True)

df_apache_valid = pd.read_csv(
    filepath_or_buffer="../intermediary_data/apache_aps_var_results_valid_mortality.csv"
)
df_apache_valid = df_apache_valid.drop(columns=["Unnamed: 0", "Unnamed: 0_x", "Unnamed: 0_y"])
df_apache_valid.head(n=3)

patient_stays_valid: list = df_apache_valid["patientunitstayid"].tolist()

vp_apache_path = Path(INTER_DATA_PATH) / Path("vital_periodic_with_apache_all.parquet")

if vp_apache_path.is_file():
    df_vp_apache_valid = pd.read_parquet(vp_apache_path, engine="pyarrow")
    # df_vp_apache_valid = pd.read_csv(vp_apache_path, engine="pyarrow")
    logger.info(f"Loading data from local file")
else:
    logger.error(f"File not found")
    df_vp_apache_valid = None


df_vp_apache_valid_grouped = df_vp_apache_valid.groupby("patientunitstayid")
df_vp_apache_valid_grouped.head()


unique_stays = df_vp_apache_valid["patientunitstayid"].unique()
print(
    f"We got {len(unique_stays)} unique patientunitids with an Apache score and the vitalperiodic table in the dataset"
)


print(f"Number of patientunitstayids with valid apache predictions {len(patient_stays_valid)}")
print(
    f"{len(patient_stays_valid) - len(unique_stays)} patientunitstayids are missing in the vitalperiodic table"
)

vp_parameters = list()
for column in df_vp_apache_valid.columns:
    if (
        column == "vitalperiodicid"
        or column == "patientunitstayid"
        or column == "observationoffset"
        or column == ""
    ):
        pass
    else:
        vp_parameters.append(column)


def generate_parameter_stats(df_vp_apache_valid_grouped, vp_parameters):
    measurement_id_count = dict()

    for group_id, group_df in df_vp_apache_valid_grouped:
        stay_id = group_id
        number_measurements = len(group_df)
        measurement_id_count.update({stay_id: number_measurements})

        parameter_sample_count = {"patientunitstayid": stay_id}
        parameter_coverage_ratio = {"patientunitstayid": stay_id}
        parameter_missing_count = {"patientunitstayid": stay_id}

        for param in vp_parameters:
            count = group_df[param].count()
            coverage_ratio = count / number_measurements

            count_missing = number_measurements - count

            parameter_sample_count[param] = count
            parameter_coverage_ratio[param] = coverage_ratio
            parameter_missing_count[param] = count_missing

        yield parameter_sample_count, parameter_coverage_ratio, parameter_missing_count


parameter_sample_count_list_all = []
parameter_missing_count_list_all = []
parameter_coverage_ratio_list_all = []


for sample_count, coverage_ratio, missing_count in generate_parameter_stats(
    df_vp_apache_valid_grouped, vp_parameters
):
    parameter_sample_count_list_all.append(sample_count)
    parameter_missing_count_list_all.append(missing_count)
    parameter_coverage_ratio_list_all.append(coverage_ratio)

parameter_sample_count_df = pd.DataFrame(data=parameter_sample_count_list_all)
parameter_missing_count_df = pd.DataFrame(data=parameter_missing_count_list_all)
parameter_coverage_ratio_df = pd.DataFrame(data=parameter_coverage_ratio_list_all)


parameter_sample_count_path = Path(INTER_DATA_PATH) / Path("vital_periodic_all_sample_count.csv")
parameter_missing_count_path = Path(INTER_DATA_PATH) / Path("vital_periodic_all_missing_count.csv")
parameter_coverage_count_path = Path(INTER_DATA_PATH) / Path(
    "vital_periodic_all_coverage_count.csv"
)

if entire_dataset:
    parameter_sample_count_df.to_csv(path_or_buf=parameter_sample_count_path)
    parameter_missing_count_df.to_csv(path_or_buf=parameter_missing_count_path)
    parameter_coverage_ratio_df.to_csv(path_or_buf=parameter_coverage_count_path)


number_stays = parameter_coverage_ratio_df.shape[0]

fig_sample, ax_sample = plt.subplots(1, 1, figsize=(15, 15))


sns.heatmap(
    parameter_sample_count_df.drop(["patientunitstayid"], axis=1), ax=ax_sample, annot=False
)

ax_sample.collections[0].colorbar.set_label("Number of Samples")
plt.title(f"{number_stays} ICU stays; number of samples per variable")

filename_corr = "number_of_samples_per_variable_heatmap"
notebook_name = "vitalperiodic_exploration"
plot_file_corr_path = Path("plot_outputs") / Path(notebook_name) / Path(filename_corr)

plt.savefig(str(plot_file_corr_path) + ".png", dpi=300, bbox_inches="tight")

fig_cov_rat, ax_cov_rat = plt.subplots(1, 1, figsize=(15, 15))

sns.heatmap(
    parameter_coverage_ratio_df.drop(["patientunitstayid"], axis=1), ax=ax_cov_rat, annot=False
)

ax_cov_rat.collections[0].colorbar.set_label("Ratio")
plt.title(f"{number_stays} ICU stays; #samples/#measurements")


filename_corr = "ratio_samples_measurement_per_variable_heatmap"
notebook_name = "vitalperiodic_exploration"
plot_file_corr_path = Path("plot_outputs") / Path(notebook_name) / Path(filename_corr)

plt.savefig(str(plot_file_corr_path) + ".png", dpi=300, bbox_inches="tight")

fig_missing, ax_missing = plt.subplots(1, 1, figsize=(15, 15))

sns.heatmap(
    parameter_missing_count_df.drop(["patientunitstayid"], axis=1), ax=ax_missing, annot=False
)

ax_missing.collections[0].colorbar.set_label("Number")
plt.title(f"{number_stays} ICU stays; #missing values")


filename_corr = "number_missing_values_per_variable_heatmap"
notebook_name = "vitalperiodic_exploration"
plot_file_corr_path = Path("plot_outputs") / Path(notebook_name) / Path(filename_corr)

plt.savefig(str(plot_file_corr_path) + ".png", dpi=300, bbox_inches="tight")
