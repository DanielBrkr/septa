import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from machine_intelligence.utilities.data_handling import load_parquet_table
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import polars as pl
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import pickle


from machine_intelligence.utilities.dataset_utils import align_to_grid, add_overall_sample_idx_to_df
from machine_intelligence.utilities.dataset_utils_polars import (
    align_to_grid_polars_pure,
    add_overall_sample_idx_to_df_polars,
    SampleCount,
)

from machine_intelligence.utilities.data_quality import (
    get_parameter_coverage_ratio_df,
)


def time_series_placeholder_data(
        df_dataset_ts, features: list[str], debug_dataset: bool = True
) -> [pd.DataFrame]:
    patient_ids = df_dataset_ts["patientunitstayid"].unique().tolist()

    features_and_needed = [
        *features,
        "patientunitstayid",
        "time_offset",
        "sepsis_present",
    ]
    df_dataset_ts_reduced = df_dataset_ts.loc[:, features_and_needed]

    df_dataset_ts_reduced = df_dataset_ts_reduced[
        df_dataset_ts_reduced["patientunitstayid"].isin(patient_ids)
    ].copy()

    df_dataset_ts_sepsis = df_dataset_ts[
        df_dataset_ts["sepsis_patient"] == True]  # noqa: E712, supposed solution changes function

    patient_ids_sepsis = df_dataset_ts_sepsis["patientunitstayid"].unique().tolist()

    df_dataset_ts_no_sepsis = df_dataset_ts[
        df_dataset_ts["sepsis_patient"] == False]  # noqa: E712, supposed solution changes function

    patient_ids_no_sepsis = df_dataset_ts_no_sepsis["patientunitstayid"].unique().tolist()

    if debug_dataset:
        df_dataset_ts_debug = df_dataset_ts_reduced.loc[
            df_dataset_ts_reduced["patientunitstayid"].isin(
                [*patient_ids_no_sepsis[0:10], *patient_ids_sepsis[0:10]]
            )
        ]

        df_train = df_dataset_ts_debug.loc[
            df_dataset_ts_debug["patientunitstayid"].isin(
                [*patient_ids_no_sepsis[0:5], *patient_ids_sepsis[0:5]]
            )
        ]

        df_test = df_dataset_ts_debug.loc[
            df_dataset_ts_debug["patientunitstayid"].isin(
                [*patient_ids_no_sepsis[0:5], *patient_ids_sepsis[0:5]]
            )
        ]
    else:
        df_train = df_dataset_ts_reduced.loc[
            df_dataset_ts_reduced["patientunitstayid"].isin(patient_ids[0:100])
        ]
        df_test = df_dataset_ts_reduced.loc[
            df_dataset_ts_reduced["patientunitstayid"].isin(patient_ids[101:200])
        ]

    return df_train, df_test


def generate_stratified_ts_datasets(
        df_dataset, features: list[str] | None, debug: bool, save_sets: bool
) -> [pd.DataFrame]:
    """Deals currenlty only with time series and not yet with static variables, has to be implemented yet!"""
    random_state: int = 42

    df_stratification = (
        df_dataset.loc[:, ["patientunitstayid", "sepsis_patient"]].copy().drop_duplicates()
    )
    df_stratification["sepsis_patient"] = df_stratification[["sepsis_patient"]].astype("boolean")
    df_stratification.head()

    if debug_strat := False:
        patient_ids_all = df_stratification["patientunitstayid"].unique().tolist()
        patient_ids_debug = patient_ids_all[::50]
        df_stratification = df_stratification.loc[
            df_stratification["patientunitstayid"].isin(patient_ids_debug)
        ]

    x_initial_split = df_stratification["patientunitstayid"].reset_index(drop=True)
    y_initial_split = df_stratification["sepsis_patient"].reset_index(drop=True)

    sss_train_test = StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, test_size=0.2, random_state=random_state
    )
    sss_train_val = StratifiedShuffleSplit(
        n_splits=1, train_size=0.75, test_size=0.25, random_state=random_state
    )

    #  this should appease the lsp
    train_val_indices = None
    test_indices = None
    train_indices = None
    val_indices = None

    for i, (train_val, test) in enumerate(sss_train_test.split(x_initial_split, y_initial_split)):
        train_val_indices = train_val
        test_indices = test

    x_train_val = x_initial_split.loc[train_val_indices].reset_index(drop=True)
    x_test = x_initial_split.loc[test_indices].reset_index(drop=True)

    y_train_val = y_initial_split.loc[train_val_indices].reset_index(drop=True)
    y_test = y_initial_split.loc[test_indices].reset_index(drop=True)

    for i, (train, val) in enumerate(sss_train_val.split(x_train_val, y_train_val)):
        train_indices = train
        val_indices = val

    x_train = x_train_val.loc[train_indices]
    x_val = x_train_val.loc[val_indices]

    # we end up with train, val, test ratios of 0.6, 0.2, 0.2
    total_samples = len(x_initial_split)

    logger.info(
        f"Total samples: {total_samples}, Train, Validation, Test - Ratios: "
        f"{len(x_train) / total_samples}, {len(x_val) / total_samples} , {len(x_test) / total_samples}"
    )

    if features is None:
        df_train_strat = df_dataset.loc[df_dataset["patientunitstayid"].isin([*x_train]), :]
        df_val_strat = df_dataset.loc[df_dataset["patientunitstayid"].isin([*x_val]), :]
        df_test_strat = df_dataset.loc[df_dataset["patientunitstayid"].isin([*x_test]), :]

    else:
        mandatory_columns = ["patientunitstayid", "time_offset", "sepsis_patient", "sepsis_present"]
        considered_columns = [*mandatory_columns, *features]

        df_train_strat = df_dataset.loc[
            df_dataset["patientunitstayid"].isin([*x_train]), considered_columns
        ]
        df_val_strat = df_dataset.loc[
            df_dataset["patientunitstayid"].isin([*x_val]), considered_columns
        ]
        df_test_strat = df_dataset.loc[
            df_dataset["patientunitstayid"].isin([*x_test]), considered_columns
        ]

    df_train_strat["sepsis_patient"] = df_train_strat[["sepsis_patient"]].astype("boolean")

    df_val_strat["sepsis_patient"] = df_val_strat[["sepsis_patient"]].astype("boolean")

    df_test_strat["sepsis_patient"] = df_test_strat[["sepsis_patient"]].astype("boolean")

    # asserting that there's no patient id bleed over

    train_ids = set(df_train_strat["patientunitstayid"].unique().tolist())
    val_ids = set(df_val_strat["patientunitstayid"].unique().tolist())
    test_ids = set(df_test_strat["patientunitstayid"].unique().tolist())

    assert len(train_ids & val_ids & test_ids) == 0

    train_label_stats = df_train_strat.drop_duplicates(subset="patientunitstayid")[
        "sepsis_patient"
    ].value_counts()
    val_label_stats = df_val_strat.drop_duplicates(subset="patientunitstayid")[
        "sepsis_patient"
    ].value_counts()
    test_label_stats = df_test_strat.drop_duplicates(subset="patientunitstayid")[
        "sepsis_patient"
    ].value_counts()

    # This could be reused for the weights of the cross entropy loss
    # but we would need to calculate the overall class imbalance
    train_label_ratio = train_label_stats.loc[True] / train_label_stats.loc[False]
    val_label_ratio = val_label_stats.loc[True] / val_label_stats.loc[False]
    test_label_ratio = test_label_stats.loc[True] / test_label_stats.loc[False]

    logger.info(
        f"Train Dataset Sepsis (False vs. True) Ratio: {train_label_ratio:.2%}, #Sepsis Cases: {train_label_stats.loc[True]}"
    )
    logger.info(
        f"Valid Dataset Sepsis (False vs. True) Ratio: {val_label_ratio:.2%}, #Sepsis Cases: {val_label_stats.loc[True]}"
    )
    logger.info(
        f"Test Dataset Sepsis (False vs. True) Ratio: {test_label_ratio:.2%}, #Sepsis Cases: {test_label_stats.loc[True]}"
    )

    if save_sets:
        today = datetime.today().strftime("%Y-%m-%d")

    current_path: Path = Path.cwd()
    project_root: Path = current_path.parents[1]
    datasets_path: Path = project_root / Path("datasets")
    data_save_path: Path = datasets_path / Path("splitted") / Path(today)

    data_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"{current_path = }")
    logger.info(f"{data_save_path = }")

    return df_train_strat, df_val_strat, df_test_strat


def generate_stratified_ts_datasets_via_sample(
        df_dataset, features: list[str] | None, sample_idx_col: str, target_col: str, debug: bool, save_sets: bool
) -> [pd.DataFrame]:
    """Deals currenlty only with time series and not yet with static variables, has to be implemented yet!"""
    random_state: int = 42

    df_stratification = (
        df_dataset.loc[:, [sample_idx_col, target_col]].copy().drop_duplicates()
    )
    df_stratification[target_col] = df_stratification[[target_col]].astype("boolean")
    df_stratification.head()

    if debug_strat := False:
        patient_ids_all = df_stratification["patientunitstayid"].unique().tolist()
        patient_ids_debug = patient_ids_all[::50]
        df_stratification = df_stratification.loc[
            df_stratification["patientunitstayid"].isin(patient_ids_debug)
        ]

    x_initial_split = df_stratification[sample_idx_col].reset_index(drop=True)
    y_initial_split = df_stratification[target_col].reset_index(drop=True)

    sss_train_test = StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, test_size=0.2, random_state=random_state
    )
    sss_train_val = StratifiedShuffleSplit(
        n_splits=1, train_size=0.75, test_size=0.25, random_state=random_state
    )

    #  this should appease the lsp
    train_val_indices = None
    test_indices = None
    train_indices = None
    val_indices = None

    for i, (train_val, test) in enumerate(sss_train_test.split(x_initial_split, y_initial_split)):
        train_val_indices = train_val
        test_indices = test

    x_train_val = x_initial_split.loc[train_val_indices].reset_index(drop=True)
    x_test = x_initial_split.loc[test_indices].reset_index(drop=True)

    y_train_val = y_initial_split.loc[train_val_indices].reset_index(drop=True)
    y_test = y_initial_split.loc[test_indices].reset_index(drop=True)

    for i, (train, val) in enumerate(sss_train_val.split(x_train_val, y_train_val)):
        train_indices = train
        val_indices = val

    x_train = x_train_val.loc[train_indices]
    x_val = x_train_val.loc[val_indices]

    # we end up with train, val, test ratios of 0.6, 0.2, 0.2
    total_samples = len(x_initial_split)

    logger.info(
        f"Total samples: {total_samples}, Train, Validation, Test - Ratios: "
        f"{len(x_train) / total_samples}, {len(x_val) / total_samples} , {len(x_test) / total_samples}"
    )

    if features is None:
        df_train_strat = df_dataset.loc[df_dataset[sample_idx_col].isin([*x_train]), :]
        df_val_strat = df_dataset.loc[df_dataset[sample_idx_col].isin([*x_val]), :]
        df_test_strat = df_dataset.loc[df_dataset[sample_idx_col].isin([*x_test]), :]

    else:
        mandatory_columns = ["patientunitstayid", "time_offset", "sepsis_patient", "sepsis_present"]
        considered_columns = [*mandatory_columns, *features]

        df_train_strat = df_dataset.loc[
            df_dataset["patientunitstayid"].isin([*x_train]), considered_columns
        ]
        df_val_strat = df_dataset.loc[
            df_dataset["patientunitstayid"].isin([*x_val]), considered_columns
        ]
        df_test_strat = df_dataset.loc[
            df_dataset["patientunitstayid"].isin([*x_test]), considered_columns
        ]

    df_train_strat[target_col] = df_train_strat[[target_col]].astype("boolean")

    df_val_strat[target_col] = df_val_strat[[target_col]].astype("boolean")

    df_test_strat[target_col] = df_test_strat[[target_col]].astype("boolean")

    # asserting that there's no patient id bleed over

    train_ids = set(df_train_strat[target_col].unique().tolist())
    val_ids = set(df_val_strat[target_col].unique().tolist())
    test_ids = set(df_test_strat[target_col].unique().tolist())

    assert len(train_ids & val_ids & test_ids) == 0

    train_label_stats = df_train_strat.drop_duplicates(subset=sample_idx_col)[
        target_col
    ].value_counts()
    val_label_stats = df_val_strat.drop_duplicates(subset=sample_idx_col)[
        target_col
    ].value_counts()
    test_label_stats = df_test_strat.drop_duplicates(subset=sample_idx_col)[
        target_col
    ].value_counts()

    # This could be reused for the weights of the cross entropy loss
    # but we would need to calculate the overall class imbalance
    train_label_ratio = train_label_stats.loc[True] / train_label_stats.loc[False]
    val_label_ratio = val_label_stats.loc[True] / val_label_stats.loc[False]
    test_label_ratio = test_label_stats.loc[True] / test_label_stats.loc[False]

    logger.info(
        f"Train Dataset Sepsis (False vs. True) Ratio: {train_label_ratio:.2%}, #Sepsis Cases: {train_label_stats.loc[True]}"
    )
    logger.info(
        f"Valid Dataset Sepsis (False vs. True) Ratio: {val_label_ratio:.2%}, #Sepsis Cases: {val_label_stats.loc[True]}"
    )
    logger.info(
        f"Test Dataset Sepsis (False vs. True) Ratio: {test_label_ratio:.2%}, #Sepsis Cases: {test_label_stats.loc[True]}"
    )

    if save_sets:
        today = datetime.today().strftime("%Y-%m-%d")

    current_path: Path = Path.cwd()
    project_root: Path = current_path.parents[1]
    datasets_path: Path = project_root / Path("datasets")
    data_save_path: Path = datasets_path / Path("splitted") / Path(today)

    data_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"{current_path = }")
    logger.info(f"{data_save_path = }")

    return df_train_strat, df_val_strat, df_test_strat


def align_and_add_sample_index_individual(
        df_split_in: pd.DataFrame,
) -> pd.DataFrame:
    # raw_dataset_dfs: list[pd.DataFrame] = [df_train_in, df_val_in, df_test_in]

    icu_offset_df: pd.DataFrame = df_split_in[df_split_in["time_offset"] >= pd.to_timedelta(0)]

    logger.info("Removing time stamps prior to the ICU stay done!")

    tqdm.pandas(desc="Aligning dataframe")

    aligned_df: pd.DataFrame = (
        icu_offset_df.groupby("patientunitstayid", group_keys=False)[icu_offset_df.columns]
        .progress_apply(lambda x: align_to_grid(x))
        .reset_index(drop=True)
    )

    logger.info("Aligning done!")

    df_split_indexed: pd.DataFrame = add_overall_sample_idx_to_df(df_for_dataloader=aligned_df)

    logger.info("Done adding sample index to set!")

    return df_split_indexed


def align_individual_df(df: pd.DataFrame) -> pd.DataFrame:
    icu_offset_df: pd.DataFrame = df[df["time_offset"] >= pd.to_timedelta(0)]

    logger.info("Removing time stamps prior to the ICU stay done!")

    tqdm.pandas(desc="Aligning dataframe")

    for column in icu_offset_df.select_dtypes(include=["category"]).columns:
        icu_offset_df.loc[:, column] = icu_offset_df.loc[:, column].astype(int)

    icu_offset_df = pl.from_pandas(icu_offset_df)

    aligned_df: pl.DataFrame = icu_offset_df.group_by("patientunitstayid").map_groups(
        function=lambda x: align_to_grid_polars_pure(x)
    )

    aligned_df: pd.DataFrame = aligned_df.to_pandas()

    logger.info("Aligning done!")

    return aligned_df


def add_sample_indices_to_df(df: pd.DataFrame, input_pred_window_h, stage: str) -> pd.DataFrame:
    df_indexed: pd.DataFrame = add_overall_sample_idx_to_df_polars(
        df_for_dataloader=df, input_pred_window_h=input_pred_window_h, stage=stage
    )

    logger.info("Done adding sample index to set!")

    return df_indexed


def drop_samples_after_onset(
        df_group: pd.DataFrame.groupby, df_patient_onset_indices, sample_index_col
) -> pd.DataFrame:
    onset_patient_ids = df_patient_onset_indices["patientunitstayid"].tolist()
    patient_id = df_group["patientunitstayid"].iloc[-1]

    if patient_id in onset_patient_ids:
        onset_sample_idx = df_patient_onset_indices.loc[
            df_patient_onset_indices["patientunitstayid"] == patient_id, sample_index_col
        ].item()
        df_to_keep = df_group[df_group[sample_index_col] <= onset_sample_idx]

        df_to_keep["sepsis_sample"] = False
        df_to_keep.loc[df_to_keep[sample_index_col] == onset_sample_idx, "sepsis_sample"] = True

        # Todo: Sample index colum zu True setzen

    else:
        df_to_keep = df_group
        df_to_keep["sepsis_sample"] = False

    return df_to_keep


def remove_ongoing_sepsis_sample(
        df_preindexed: pd.DataFrame, sample_index_column: str
) -> pd.DataFrame:
    logger.info("Starting to remove ongoing sepsis samples !")
    sepsis_df = df_preindexed[df_preindexed["sepsis_present"] == True]  # noqa: E712
    sepsis_onset_sample_indices = sepsis_df.drop_duplicates("patientunitstayid")

    df_wo_ongoing = df_preindexed.groupby(by="patientunitstayid", group_keys=False).progress_apply(
        lambda x: drop_samples_after_onset(
            df_group=x,
            df_patient_onset_indices=sepsis_onset_sample_indices,
            sample_index_col=sample_index_column,
        )
    )

    logger.info("Removing ongoing sepsis samples done !")

    return df_wo_ongoing


def shift_target_and_drop_last_sample(
        df: pd.DataFrame, prediction_horizon_h: int = 6, time_grid_freq: str = "5min"
) -> pd.DataFrame:
    def shift_and_drop_group(df_group, prediction_horizon_h, time_grid_freq: str = "5min"):
        prediction_horizon = pd.to_timedelta(prediction_horizon_h, unit="h")
        prediction_target_shifted = "sepsis_sample_shifted"

        prediction_horizon_time_steps_number = int(
            prediction_horizon / pd.to_timedelta(time_grid_freq)
        )

        if df_group["sepsis_patient"].iloc[0] == True:
            df_group[prediction_target_shifted] = df_group["sepsis_sample"].shift(
                - prediction_horizon_time_steps_number
            )

            df_group[prediction_target_shifted] = df_group[prediction_target_shifted].astype(
                "boolean"
            )

        else:
            df_group[prediction_target_shifted] = df_group["sepsis_sample"]

        df_group = df_group.dropna(subset=[prediction_target_shifted])

        return df_group

    df_shifted = df.groupby(by="patientunitstayid", group_keys=False).apply(
        lambda x: shift_and_drop_group(df_group=x, prediction_horizon_h=prediction_horizon_h)
    )

    return df_shifted


def resample_to_sepsis_prevalence(df: pd.DataFrame, sepsis_prevalence_ratio: float,
                                  sample_index_column: int) -> pd.DataFrame:
    # replacement false prevents duplicates majority class samples
    rus = RandomUnderSampler(sampling_strategy=sepsis_prevalence_ratio, random_state=42, replacement=False)

    df_sample_label_indices = (
        df[[sample_index_column, "sepsis_sample_shifted"]].drop_duplicates()
    )

    df_sample_label_indices["workaround"] = 0

    df_rus_sample_idx, df_rus_label = rus.fit_resample(
        df_sample_label_indices[["workaround", sample_index_column]],
        df_sample_label_indices["sepsis_sample_shifted"],
    )

    df_resampled = df[df[sample_index_column].isin(df_rus_sample_idx[sample_index_column])]

    logger.info(f"Resampling Result: {sorted(Counter(df_rus_label).items())}")

    return df_resampled


def reindex_samplex_indices(df: pd.DataFrame, sample_index_column: str):
    def assign_new_index(df_group, sample_idx: SampleCount):
        df_group[sample_index_column] = sample_idx.value()
        sample_idx.increment()

        return df_group

    logger.info("Reindexing sample indicies!")
    SampleIndex = SampleCount()
    df_reindexed = df.groupby(by=sample_index_column, group_keys=False).progress_apply(
        lambda x: assign_new_index(df_group=x, sample_idx=SampleIndex)
    )

    SampleIndex.reset()
    logger.info("Reindexing sample indicies done!")

    return df_reindexed


def align_and_add_sample_index_individual_polars(
        df_split_in: pd.DataFrame,
) -> pd.DataFrame:
    # raw_dataset_dfs: list[pd.DataFrame] = [df_train_in, df_val_in, df_test_in]

    icu_offset_df: pd.DataFrame = df_split_in[df_split_in["time_offset"] >= pd.to_timedelta(0)]

    logger.info("Removing time stamps prior to the ICU stay done!")

    tqdm.pandas(desc="Aligning dataframe")

    # aligned_df: pd.DataFrame = (
    #     icu_offset_df.groupby("patientunitstayid", group_keys=False)[icu_offset_df.columns]
    #     .progress_apply(lambda x: align_to_grid_polars(x))
    #     .reset_index(drop=True)
    # )
    #

    # for column in icu_offset_df.select_dtypes(include=['category']).columns:
    #     icu_offset_df.loc[:,column] = icu_offset_df.loc[:,column].astype(str)

    for column in icu_offset_df.select_dtypes(include=["category"]).columns:
        icu_offset_df.loc[:, column] = icu_offset_df.loc[:, column].astype(int)

    icu_offset_df = pl.from_pandas(icu_offset_df)

    aligned_df: pl.DataFrame = icu_offset_df.group_by("patientunitstayid").map_groups(
        function=lambda x: align_to_grid_polars_pure(x)
    )

    aligned_df: pd.DataFrame = aligned_df.to_pandas()

    logger.info("Aligning done!")

    df_split_indexed: pd.DataFrame = add_overall_sample_idx_to_df_polars(
        df_for_dataloader=aligned_df
    )

    logger.info("Done adding sample index to set!")

    return df_split_indexed


def align_and_add_sample_index(
        df_train_in: pd.DataFrame,
        df_val_in: pd.DataFrame,
        df_test_in: pd.DataFrame,
) -> [pd.DataFrame]:
    raw_dataset_dfs: list[pd.DataFrame] = [df_train_in, df_val_in, df_test_in]

    icu_offset_dfs: list[pd.DataFrame] = [
        frame[frame["time_offset"] >= pd.to_timedelta(0)] for frame in raw_dataset_dfs
    ]
    logger.info("Removing time stamps prior to the ICU stay done!")

    # this is a fix needed for dask (possibly)
    icu_offset_dfs: list[pd.DataFrame] = [frame.reset_index() for frame in raw_dataset_dfs]

    tqdm.pandas(desc="Aligning dataframes")

    aligned_dfs: [pd.DataFrame] = [
        frame.groupby("patientunitstayid", group_keys=False)[frame.columns]
        .progress_apply(lambda x: align_to_grid(x))
        .reset_index(drop=True)
        for frame in icu_offset_dfs
    ]

    logger.info("Aligning done!")

    indexed_dfs: [pd.DataFrame] = [
        add_overall_sample_idx_to_df(df_for_dataloader=frame) for frame in aligned_dfs
    ]

    logger.info("Done adding sample index to each dataset!")

    df_train_indexed, df_val_indexed, df_test_indexed = indexed_dfs

    return df_train_indexed, df_val_indexed, df_test_indexed


def scale_ts_variables(df_train, df_val, df_test, features_to_scale, method: str = "MinMax") -> [pd.DataFrame]:
    logger.info("Starting scaling of variables!")

    if method == "MinMax":
        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    else:
        scaler = StandardScaler()

    scaler.fit(X=df_train.loc[:, features_to_scale])

    logger.info("Fitting of scaler done, starting to transform columns !")

    df_train_in_stats = df_train.describe()

    logger.info(f"Stats of the train set before scaling: \n {df_train_in_stats = }")

    df_train.loc[:, features_to_scale] = scaler.transform(df_train.loc[:, features_to_scale])
    df_val.loc[:, features_to_scale] = scaler.transform(df_val.loc[:, features_to_scale])
    df_test.loc[:, features_to_scale] = scaler.transform(df_test.loc[:, features_to_scale])

    df_train_scaled_stats = df_train.describe()
    logger.info(f"Stats of the train set after scaling: \n {df_train_scaled_stats = }")

    with open("scaler_train.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return df_train, df_val, df_test


def dump_low_density_patient_stays(
        df: pd.DataFrame, density_threshold: float = 0.25
) -> pd.DataFrame:
    df_sorted = df.sort_values(by=["patientunitstayid", "time_offset"])
    logger.debug(f"{df_sorted.patientunitstayid.is_monotonic_increasing =}")
    df_grouped = df_sorted.groupby(by="patientunitstayid", group_keys=False)

    df_variable_densities = get_parameter_coverage_ratio_df(
        df_grouped=df_grouped, parameters=df.columns
    )

    df_variable_densities.index = df["patientunitstayid"].astype("int").unique().tolist()

    df_patients_to_dump = df_variable_densities.loc[
        (df_variable_densities["heartrate"] < density_threshold)
    ]  # or (df_variable_densities["heartrate"] > dump_threshold)]

    patient_ids_dump = df_patients_to_dump.index.tolist()

    df["patientunitstayid"] = df["patientunitstayid"].astype("int")

    df_remaining = df[~df["patientunitstayid"].isin(patient_ids_dump)]

    return df_remaining


def determine_dataset_stats_stats(df_train, df_val, df_test, sample_idx_col: str, target_col: str, stage: str,
                                  ) -> [pd.DataFrame]:
    train_sepsis_stats = df_train.drop_duplicates(subset=sample_idx_col)[
        target_col
    ].value_counts()

    val_sepsis_stats = df_val.drop_duplicates(subset=sample_idx_col)[
        target_col
    ].value_counts()
    test_sepsis_stats = df_test.drop_duplicates(subset=sample_idx_col)[
        target_col
    ].value_counts()

    logger.warning(f"[{stage}]: {target_col = }  {train_sepsis_stats = }")
    logger.warning(f"[{stage}]: {target_col = } {val_sepsis_stats = }")
    logger.warning(f"[{stage}]: {target_col = } {test_sepsis_stats = }")

    return train_sepsis_stats, val_sepsis_stats, test_sepsis_stats


if __name__ == "__main__":
    script_path = Path.cwd()
    project_root = script_path.parents[1]
    # dataset_ts_path: Path = project_root / Path("datasets/raw_and_shitty/dataset_ts_only.parquet")
    dataset_ts_path: Path = project_root / Path("datasets/raw_and_shitty/dataset_ts_a.parquet")

    df_dataset_ts = load_parquet_table(
        parquet_path=dataset_ts_path, verbose=False, backend="numpy_nullable"
    )

    df_dataset_ts.head()
    df_dataset_ts.info()

    # patient_ids = df_dataset_ts["patientunitstayid"].unique().tolist()

    # Todo: Make sequential, such that data can be saved in between and we don't hog the all the RAM all the time
    debug_mode: bool = False
    df_train, df_val, df_test = generate_stratified_ts_datasets(
        # df_dataset=df_dataset_ts, features=["heartrate"], debug=debug_mode, save_sets=True
        df_dataset=df_dataset_ts,
        features=None,
        debug=debug_mode,
        save_sets=True,
    )
    df_train_idx, df_val_idx, df_test_idx = align_and_add_sample_index(
        df_train_in=df_train,
        df_val_in=df_val,
        df_test_in=df_test,
        save_sets=False,
        debug=debug_mode,
    )

    print(df_train_idx.head())
