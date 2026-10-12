import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset


class SepsisPredictionDataset(Dataset):
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
            transform=None
    ):
        self.df_time_series: pd.DataFrame = dataframe_time_series
        self.df_static: pd.DataFrame = dataframe_static

        self.target: str = target
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

        self.transform = transform

        # ------- time series data

        print("-" * 50, " Time Series Data ", "-" * 50)
        # logger.info(f"{features = }")
        # logger.info(f"{self.features_wo_aux = }")

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

        # -------- static data
        print("-" * 50, " Static Data ", "-" * 50)

        self.df_static = self.df_static.replace(to_replace=np.nan, value=float(0))
        # print(self.df_static.info())

    def __len__(self) -> int:
        return len(self.df_time_series.index.unique())

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        # time series data
        df_sample_time_series: pd.DataFrame = self.df_time_series.loc[idx]

        # Refactor: Workaround to remove skipped samples (which are too short)
        if isinstance(df_sample_time_series, pd.Series):
            df_sample_time_series = df_sample_time_series.to_frame().T

        df_sample_features_time_series = df_sample_time_series[self.features_wo_aux]

        if self.transform:
            df_sample_features_time_series = self.transform(df_sample_features_time_series)

        df_sample_labels = df_sample_time_series[self.target]

        features_sample_time_series = torch.tensor(
            df_sample_features_time_series.to_numpy(dtype=np.float32)
        ).float()

        # -------- static data

        # we need the patient id to find the static variables for that patient
        patient_id = df_sample_time_series.iloc[-1]["patientunitstayid"]

        df_sample_static = self.df_static.loc[
            self.df_static["patientunitstayid"] == patient_id,
        ].drop("patientunitstayid", axis=1)

        # Todo: Proper imputation on dataset value or somewhere else
        if len(df_sample_static) == 0:
            features_static = torch.zeros([1, len(df_sample_static.columns)]).float()
        else:
            features_static = torch.tensor(df_sample_static.to_numpy(dtype=np.float32)).float()

        features_static = features_static.squeeze()

        target_sample_value = int(df_sample_labels.iloc[0])
        target_sample = torch.tensor(target_sample_value).float()

        return features_sample_time_series, features_static, target_sample

    def get_labels(self):

        self.label_list = self.df_time_series.reset_index(drop=False).drop_duplicates(subset=self.sample_index_column_name)[self.target].reset_index(drop=True).to_list()
        return self.label_list
