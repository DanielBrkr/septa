import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
import time
from machine_intelligence.utilities.data_handling import load_parquet_table
from machine_intelligence.utilities.dataset_utils import (
    shift_and_fill_prediction_target,
)


"""This doesn't work as expected unfortunately! Keeping the old dataset/dataloader"""
# import os
# os.environ["POLARS_MAX_THREADS"] = "1"

import polars as pl

pl.Config().set_tbl_cols(n=-1)

class FixWinPredictionDataset(Dataset):
    def __init__(
        self,
        dataframe_time_series,
        dataframe_static,
        target: str,
        features: list[str],
        features_static_cat: list[str] = None,
        features_static_num: list[str] = None,
        sample_index_column_name: str = "sample_idx_6_h",
        auxillary_columns: list[str] = None,
        input_window_length: int = 6,
        prediction_horizon: int = 6,
    ):
        self.df_time_series: pd.DataFrame = dataframe_time_series
        # self.dataframe_static: pd.DataFrame = dataframe_static.loc[
        #     :, ["patientunitstayid", "gender", "admissionweight", "ethnicity"]
        # ]
        #

        self.df_static: pd.DataFrame = dataframe_static

        self.target: str = target
        self.target_shifted: str = f"{target}_shifted"
        self.sample_index_column_name: str = sample_index_column_name

        self.features: list[str] = features
        self.features_static_cat: list[str] = features_static_cat
        self.features_static_num: list[str] = features_static_num

        self.aux_columns: list[str] = auxillary_columns

        self.features_wo_aux: list[str] = [
            x
            for x in features
            if x
            not in [
                *self.aux_columns,
                self.sample_index_column_name,
                "",
            ]
        ]

        # ------- time series data

        print("-" * 50, " Time Series Data ", "-" * 50)
        logger.info(f"{features = }")
        logger.info(f"{self.features_wo_aux = }")
        logger.info(f"{self.target = }")
        logger.info(f"{self.target_shifted = }")

        print(f"Time Series Dataframe All Columns Info: \n {dataframe_time_series.info()}")

        self.df_time_series = self.df_time_series.loc[:, [*features, *auxillary_columns, target]]
        self.df_time_series = self.df_time_series.set_index(self.sample_index_column_name)

        # polars changes the id to floats which we have to fix
        self.df_time_series["patientunitstayid"] = self.df_time_series.loc[
            :, "patientunitstayid"
        ].astype(int)

        # polars changes the type of bools which we have to fix
        self.df_time_series[["sepsis_patient", "sepsis_present"]] = self.df_time_series[
            ["sepsis_patient", "sepsis_present"]
        ].astype("boolean")

        self.df_time_series = self.df_time_series.groupby("patientunitstayid", group_keys=False)[
            self.df_time_series.columns
        ].apply(lambda x: shift_and_fill_prediction_target(x))

        # Todo: This is "data cleaning" just for test purposes, to get valid nn outputs
        self.df_time_series = self.df_time_series.replace(to_replace=np.nan, value=float(0))

        # Todo: Refactor dev imputation
        self.df_time_series[self.features_wo_aux] = self.df_time_series[
            self.features_wo_aux
        ].ffill()

        self.df_time_series = pl.from_pandas(self.df_time_series.reset_index())


        # -------- static data
        print("-" * 50, " Static Data ", "-" * 50)

        self.df_static = self.df_static.set_index("patientunitstayid")
        self.df_static = self.df_static.loc[:, [*features_static_cat, *features_static_num]]

        self.df_static_cat_transformed = pd.get_dummies(
            self.df_static.loc[:, self.features_static_cat], dummy_na=True
        ).reset_index()

        self.df_static_num = self.df_static.loc[:, self.features_static_num]

        self.df_static = pd.concat([self.df_static_cat_transformed, self.df_static_num])

        # Todo: Dev NA cleanup
        self.df_static = self.df_static.replace(to_replace=np.nan, value=float(0))
        print(self.df_static.info())

    def __len__(self) -> int:
        # return len(self.df_time_series.index.unique())
        return len(self.df_time_series.select("sample_idx_6_h").unique())

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:

        # time series data
        # df_sample_time_series: pd.DataFrame = self.df_time_series.loc[idx]
        # df_sample_time_series: pl.DataFrame = self.df_time_series.filter(pl.col("sample_idx_6_h").is_in(idx))
        df_sample_time_series: pl.DataFrame = self.df_time_series.filter(pl.col("sample_idx_6_h") == 1)
        df_sample_time_series = df_sample_time_series.to_pandas()

        # Refactor: Workaround to remove skipped samples (which are too short)
        if isinstance(df_sample_time_series, pd.Series):
            df_sample_time_series = df_sample_time_series.to_frame().T

        # df_sample_time_series = df_sample_time_series.dropna(axis=1, how="all") # workarounmd related to the all  null columns from lab tests
        # logger.debug(f"{df_sample = }")

        df_sample_features_time_series = df_sample_time_series[self.features_wo_aux]
        df_sample_labels = df_sample_time_series[self.target]

        features_sample_time_series = torch.tensor(
            df_sample_features_time_series.to_numpy(dtype=np.float32)
        ).float()

        # we don't need the label for every time step (don't delete for reference)
        # target_sample = torch.tensor(df_sample[self.target_shifted].values).float()

        # -------- static data

        # we need the patient id to find the static variables for that patient
        patient_id = df_sample_time_series.iloc[-1]["patientunitstayid"]
        #
        df_sample_static = self.df_static.loc[
            self.df_static["patientunitstayid"] == patient_id,
        ].drop("patientunitstayid", axis=1)

        # # Todo: Proper imputation on dataset value or somewhere else
        if len(df_sample_static) == 0:
            features_static = torch.zeros([1, len(df_sample_static.columns)]).float()
        else:
            features_static = torch.tensor(df_sample_static.to_numpy(dtype=np.float32)).float()

        features_static = features_static.squeeze()

        # logger.debug(f"{df_sample_static = }")
        # logger.debug(f"{features_static = }")

        target_sample_value = int(df_sample_labels.iloc[-1])
        target_sample = torch.tensor(target_sample_value).float()

        return features_sample_time_series, features_static, target_sample
        # return features_sample_time_series



if __name__ == "__main__":
    notebook_path = Path.cwd()
    project_root = notebook_path.parents[0]
    dataset_train_path: Path = project_root / Path(
        "digital-icu-deterioration-prediction/datasets/sepsis_type_b_cohort/indexed/2024-07-11/df_train_indexed_polars.parquet"
    )
    dataset_test_path: Path = project_root / Path(
        "datasets/sepsis_type_b_cohort/indexed/2024-06-24/df_test_indexed.parquet"
    )

    dataset_static_path: Path = project_root / Path(
        "digital-icu-deterioration-prediction/datasets/sepsis_type_b_cohort/static/2024-06-23/dataset_static.parquet"
    )

    df_train = load_parquet_table(dataset_train_path, backend="numpy_nullable")

    df_static = load_parquet_table(dataset_static_path, backend="numpy_nullable")

    target = "sepsis_present"

    features = list(df_train.columns.difference([target]))

    considered_features = [
        "st3",
        "noninvasivesystolic",
        "sao2",
        "temperature",
        "respiration",
        "st2",
        "st1",
    ]
    considered_features_static_cat = [
        "gender",
        "ethnicity",
        "diabetes",
        "immunosuppression",
    ]
    considered_features_static_num = ["admissionheight"]
    sample_index_column = "sample_idx_6_h"
    aux_columns = [
        "patientunitstayid",
        "vitalaperiodicid",
        "vitalperiodicid",
        "sepsis_patient",
        "time_offset",
        "sample_idx_6_h",
    ]

    train_dataset = FixWinPredictionDataset(
        dataframe_time_series=df_train,
        dataframe_static=df_static,
        target=target,
        features=considered_features,
        features_static_cat=considered_features_static_cat,
        features_static_num=considered_features_static_num,
        sample_index_column_name=sample_index_column,
        auxillary_columns=aux_columns,
    )
    train_data_loader_pinned_workers = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    train_data_loader_pinned = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        persistent_workers=True,
    )

    train_data_loader_only_persistent = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=False,
        num_workers=1,
        persistent_workers=True,
    )

    train_data_loader_barebones = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )

    train_data_loader_polars = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        # num_workers=4,
    )

    # start_time_pinned_workers = time.time()
    # for i in range(5):
    #     batch_explainer_train = next(iter(train_data_loader_pinned_workers))
    #
    # end_time_pinned_workers = time.time()

    # start_time_pinned = time.time()
    # for i in range(5):
    #     print("hello")
    #     batch_explainer_train = next(iter(train_data_loader_pinned))
    #
    # end_time_pinned = time.time()

    start_time_only_persistent = time.time()

    # for i in range(5):
    #     print("hello")
    #     batch_explainer_train = next(iter(train_data_loader_only_persistent))
    #
    # end_time_only_persistent = time.time()

    # start_time_barebones = time.time()
    # for i in range(5):
    #     print("hello")
    #     batch_explainer_train = next(iter(train_data_loader_barebones))
    #
    # end_time_barebones = time.time()

    start_time_polars = time.time()

    for i in range(5):
        print("hello")
        batch_explainer_train = next(iter(train_data_loader_polars))

    end_time_polars = time.time()

    # logger.info(
    #     f"5 Iterations of the dataloader workers pinned persistent: {end_time_pinned_workers - start_time_pinned_workers}"
    # )
    # logger.info(
    #     f"5 Iterations of the dataloader pinned persistent: {end_time_pinned - start_time_pinned}"
    # )
    # logger.info(
    #     f"5 Iterations of the dataloader persistent: {end_time_only_persistent - start_time_only_persistent}"
    # )
    # logger.info(
    #     f"5 Iterations of the dataloader barebones: {end_time_barebones - start_time_barebones}"
    # )
    logger.info(
        f"5 Iterations of the dataloader polars: {end_time_polars - start_time_polars}"
    )
