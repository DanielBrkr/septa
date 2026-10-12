import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
import polars as pl

from pathlib import Path

from machine_intelligence.utilities.data_handling import (
    load_parquet_table,
)


def count_specific_value_in_columns(df: pd.DataFrame, value) -> pd.DataFrame:
    # missing values are indicated by "-1" in some tables
    count_dict: dict = {
        column: df[df[column] == value][column].count() for column in df.columns
    }

    value_column_name: str = str(value) + "_count"
    df_count: pd.DataFrame = pd.DataFrame.from_dict(
        data=count_dict, orient="index", columns=[value_column_name]
    )
    df_count = df_count.T

    return df_count


def generator_parameter_coverage_ratio(
    df_grouped: pd.DataFrame.groupby, parameters: list
) -> dict:
    """
    Obsolete?
    Calculates the ratio between non-NA values and the number of rows, which results in a measure of
    completeness.
    Args:
        df_grouped: Dataframe grouped by the column "patientunitstayid"
        parameters: Column names for which the metric should be calculated

    Returns: A generator dictionary which contains for each "patientunitstayid" the ratio for each column
    """
    measurement_id_count: dict = dict()

    for group_id, group_df in df_grouped:
        stay_id: int = group_id
        number_measurements: int = len(group_df)
        measurement_id_count.update({stay_id: number_measurements})

        parameter_coverage_ratio: dict = {"patientunitstayid": stay_id}

        for param in parameters:
            # counts number of non-NA cells
            count: int = group_df[param].count()
            coverage_ratio: float = count / number_measurements

            if coverage_ratio > 1:
                logger.warning(
                    f"Coverage ratio above 1 on Patient Unit Stay ID {stay_id}; "
                    f"{coverage_ratio = } "
                )
            elif coverage_ratio < 0:
                logger.warning(
                    f"Coverage ratio below 0 on Patient Unit Stay ID {stay_id}; "
                    f"{coverage_ratio = } "
                )

            parameter_coverage_ratio[param] = coverage_ratio

        yield parameter_coverage_ratio
        # return parameter_coverage_ratio


def get_parameter_coverage_ratio_df(
    df_grouped: pd.DataFrame.groupby, parameters: list
) -> pd.DataFrame:
    """
    Generates a dataframe with per patient-unit-stay-id completeness/coverage ratio, however,
    it doesn't keep the patientstayid intact, replaces it with an index of ascending order.

    Args:
        df_grouped: df_grouped: Dataframe grouped by the column "patientunitstayid"
        parameters:

    Returns: A df that contains the per patientunitstayid coverage ratio in increasing order

    """

    # There's actually no benefit of using list comprehensions with generator expression expect for "readability"
    # both perform same

    # parameter_coverage_ratio_list_all: list = []
    # for coverage_ratio in tqdm(generator_parameter_coverage_ratio(df_grouped=df_grouped, parameters=parameters),
    #                           total=len(df_grouped)):
    #    parameter_coverage_ratio_list_all.append(coverage_ratio)

    parameter_coverage_ratio_list_all_gen = [
        coverage_ratio
        for coverage_ratio in tqdm(
            generator_parameter_coverage_ratio(
                df_grouped=df_grouped, parameters=parameters
            ),
            total=len(df_grouped),
            desc="Calculating coverage ratios",
        )
    ]

    parameter_coverage_ratio_df: pd.DataFrame = pd.DataFrame(
        data=parameter_coverage_ratio_list_all_gen
    )

    return parameter_coverage_ratio_df





def categorical_feature_frequency(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Gets the counts of categorical data to catch possible NA-equivalent entries in the DQ assessment
    Args:
        frame: pd.Dataframe
        table_name: The name of the table

    Returns: pd.Dataframe with structure: feature1 | count1 | feature2 | count2

    """
    df_all_string: pd.DataFrame = frame.select_dtypes(include="string")
    frequency_string_data: dict = {}
    for column in df_all_string.columns:
        frequency_string_data[column] = df_all_string[column].value_counts()

    df_all_cats: pd.DataFrame = frame.select_dtypes(include="category")
    frequency_category_data: dict = {}
    for column in df_all_cats.columns:
        frequency_category_data[column] = df_all_cats[column].value_counts()

    dfs_string_list: list[pd.DataFrame] = []
    for key in frequency_string_data.keys():
        df_string: pd.DataFrame = pd.DataFrame(frequency_string_data[key]).reset_index()
        dfs_string_list.append(df_string)

    dfs_cats_list: list[pd.DataFrame] = []
    for key in frequency_category_data.keys():
        df_cat: pd.DataFrame = pd.DataFrame(frequency_category_data[key]).reset_index()
        dfs_cats_list.append(df_cat)

    try:
        df_cats_frequency: pd.DataFrame = pd.concat(
            [*dfs_string_list, *dfs_cats_list], axis=1, ignore_index=False
        )
    except ValueError as ex:
        logger.warning(f"Probably nothing to concatenate: {ex}")
        df_cats_frequency = pd.DataFrame(
            data={
                str(table_name): [
                    "It seems that there was not categorical or string data in the dataframe"
                ]
            }
        )

    return df_cats_frequency


if __name__ == "__main__":
    dataset_path = Path(
        "/home/danielbrkr/eicu_code_dhm/intermediary_data/dataset_cleaned/sepsis-type-b-cohort/2024-07-17/dataset_ts_cleaned.parquet"
    )

    df_ds = load_parquet_table(dataset_path, backend="numpy_nullable")
    df_ds_grouped = df_ds.groupby(by="patientunitstayid")

    get_parameter_coverage_ratio_df(
        df_grouped=df_ds_grouped, parameters=["heartrate", "sao2"]
    )
