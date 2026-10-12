import pandas as pd
from loguru import logger
from functools import reduce
from tqdm.auto import tqdm

tqdm.pandas()


def remove_redundant_rows(df):
    # if column == apacheversion -> drop rows with "IV"
    pass


def remove_discard_column(df):
    pass


@logger.catch
def replace_na_equivalent_values(
    df_in: pd.DataFrame,
    value_to_replace: str or float or int,
    save_memory: bool = False,
    description: bool = False,
) -> pd.DataFrame:
    if value_to_replace != "":
        df_missingness_in: pd.DataFrame = df_in.isna().sum(axis=0).to_frame().T
        df_description_in: pd.DataFrame = df_in.describe()

        if save_memory:
            df_out = df_in.replace(value_to_replace, pd.NA, inplace=True)
        else:
            df_out = df_in.replace(value_to_replace, pd.NA, inplace=False)

        df_missingness_out: pd.DataFrame = df_out.isna().sum(axis=0).to_frame().T
        df_description_out: pd.DataFrame = df_out.describe()

        logger.info(f"{df_missingness_in = }")
        logger.info(f"{df_missingness_out = }")

        if description:
            logger.info(f"{df_description_in = } \n {df_description_out = }")

    else:
        logger.info(f"Nothing to replace {value_to_replace = }, skipping!")
        df_out = df_in

    return df_out


def remove_wrong_apache_version_rows(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the APACHE version IV rows from the dataframe, we are only interested in
    version IVa (this version is most up to date).
    Args:
        df_in: Apache Patient Results dataframe, select the right dataframe to apply this function to
        in the caller!

    Returns: The dataframe with IV rows removed

    """

    if "apacheversion" in df_in.columns:
        df_data_out = df_in[df_in["apacheversion"] != "IV"]
        logger.info("Removing duplicate rows with the wrong APACHE version!")
    else:
        df_data_out = df_in
        logger.info(
            "Table doesn't contain duplicate rows (due to APACHE IV vs. IVa), skipping ! "
        )

    return df_data_out


def set_outliers_to_na(
    df, lower_percentile=0.01, upper_percentile=0.99, skip_columns=None
):
    """
    Sets outliers in all columns of the DataFrame to pd.NA based on the given percentiles,
    except for the columns specified to be skipped.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    lower_percentile (float): The lower percentile threshold for outlier detection.
    upper_percentile (float): The upper percentile threshold for outlier detection.
    skip_columns (list): List of column names to skip for outlier detection.

    Returns:
    pd.DataFrame: The DataFrame with outliers set to pd.NA.
    """

    if skip_columns is None:
        skip_columns = []

    for column in df.columns:
        if column not in skip_columns and pd.api.types.is_numeric_dtype(df[column]):
            col_max_pre = df[column].max()
            col_min_pre = df[column].min()
            lower_bound = df[column].quantile(lower_percentile)
            upper_bound = df[column].quantile(upper_percentile)

            # Set outliers to pd.NA
            df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = (
                pd.NA
            )

            col_max_post = df[column].max()
            col_min_post = df[column].min()

            logger.info(
                f"Performed outlier removal on column: {column}, \n"
                f" {col_min_pre = }, {col_max_pre = }, \n {col_min_post = }, {col_max_post = }"
            )

    return df


def drop_unnecessary_columns(
    df_in: pd.DataFrame, dropping_flag: bool, columns_drop: list, columns_keep: list
) -> pd.DataFrame:
    if dropping_flag:
        logger.info("Dropping columns!")
        if len(columns_drop) != 0:
            df_out = df_in.drop(columns=columns_drop)
        elif len(columns_keep) != 0:
            df_out = df_in[columns_keep]
        else:
            df_out = None
            logger.error(
                f"There's something wrong {dropping_flag = } but no columns where specified!"
            )

    else:
        df_out = df_in
        logger.info(f"Won't drop any columns from table! {dropping_flag = }")

    return df_out


# obsolete
def merge_ts_tables(dfs_to_merge: list[pd.DataFrame]) -> pd.DataFrame:
    df_joined_ts = reduce(
        lambda left, right: left.merge(
            right, on=["patientunitstayid", "time_offset"], how="outer"
        ),
        dfs_to_merge,
    )

    return df_joined_ts


# obsolete
def resample_patient_timeseries(df_to_resample: pd.DataFrame) -> pd.DataFrame:
    def resample_group(group):
        resampling_freq_string: str = "5min"

        resampled_patient_sub_df = group.resample(
            rule=resampling_freq_string, on="time_offset", origin="start"
        ).last()

        resampled_patient_sub_df = resampled_patient_sub_df.reset_index()

        return resampled_patient_sub_df

    resampled_df = df_to_resample.groupby(
        "patientunitstayid", group_keys=False
    ).progress_apply(resample_group)
    resampled_df["patientunitstayid"] = resampled_df["patientunitstayid"].convert_dtypes()

    return resampled_df
