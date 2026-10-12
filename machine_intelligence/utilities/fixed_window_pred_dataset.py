import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

import time

from machine_intelligence.utilities.dataset_utils import (
    shift_and_fill_prediction_target,
    align_to_grid,
)


class FixedWindowPredictionDataset(Dataset):
    def __init__(
        self,
        dataframe,
        features: list[str],
        target: str,
        input_window_length: int = 6,
        prediction_horizon: int = 6,
    ):
        self.dataframe: pd.DataFrame = dataframe
        self.features: list[str] = features
        self.features_wo_indices: list[str] = [
            x
            for x in features
            if x not in ["patientunitstayid", "time_offset", "sepsis_present", "overall_row_id", ""]
        ]
        self.target: str = target
        self.target_shifted: str = f"{target}_shifted"

        logger.info(f"{features = }")
        logger.info(f"{target = }")
        logger.info(f"{dataframe.head()}")

        # dataset pre-processing
        # 0. Dropping rows/data with a negative time offset
        # 1. align to specified grid
        # 2. shift and forward fill prediction target accordingly
        # 3. add new sample index via that identifies a slice/chunk based on the input
        #    window size, the dataloader will iterate over this index

        self.dataframe = (
            self.dataframe.groupby("patientunitstayid", group_keys=False)[
                self.dataframe.columns
            ].apply(lambda x: shift_and_fill_prediction_target(x))
            # .reset_index(drop=True)
        )

        logger.debug(f"{self.dataframe.head()}")
        print("Hello")
        # logger.info(f"{self.dataframe.info()}")
        # logger.info(f"{self.dataframe.head()}")

    def __len__(self) -> int:
        # return self.X.shape[0]
        # return len(self.dataframe)
        return len(self.dataframe.index.unique())

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        df_sample: pd.DataFrame = self.dataframe.loc[idx]

        # Todo: This is "data cleaning" just for test purposes, to get valid nn outputs
        df_sample = df_sample.replace(to_replace=np.nan, value=float(0))
        # logger.debug(f"{df_sample = }")

        df_sample_features = df_sample[self.features_wo_indices]
        df_sample_labels = df_sample[self.target_shifted]

        features_sample = torch.tensor(df_sample_features.to_numpy(dtype=np.float32)).float()

        # we don't need the label for every time step (don't delete for reference)
        # target_sample = torch.tensor(df_sample[self.target_shifted].values).float()

        target_sample_value = int(df_sample_labels.iloc[-1])
        target_sample = torch.tensor(target_sample_value).float()

        return features_sample, target_sample


class FixedWindowPredictionTsStaticDataset(Dataset):
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
        self.df_time_series[["sepsis_patient", "sepsis_present"]] = (
            self.df_time_series[["sepsis_patient", "sepsis_present"]].astype("boolean")
        )

        self.df_time_series = self.df_time_series.groupby(
            "patientunitstayid", group_keys=False
        )[self.df_time_series.columns].apply(lambda x: shift_and_fill_prediction_target(x))

        # Todo: This is "data cleaning" just for test purposes, to get valid nn outputs
        self.df_time_series = self.df_time_series.replace(to_replace=np.nan, value=float(0))

        # Todo: Refactor dev imputation
        self.df_time_series[self.features_wo_aux] = self.df_time_series[
            self.features_wo_aux
        ].ffill()

        # -------- static data
        print("-" * 50, " Static Data ", "-" * 50)

        self.df_static = self.df_static.set_index("patientunitstayid")
        self.df_static = self.df_static.loc[:, [*features_static_cat, *features_static_num]]

        self.df_static_cat_transformed = pd.get_dummies(
            self.df_static.loc[:, self.features_static_cat], dummy_na=True
        ).reset_index()

        self.df_static_num = self.df_static.loc[:, self.features_static_num]

        self.df_static = pd.concat(
            [self.df_static_cat_transformed, self.df_static_num]
        )

        # Todo: Dev NA cleanup
        self.df_static = self.df_static.replace(to_replace=np.nan, value=float(0))
        print(self.df_static.info())

    def __len__(self) -> int:
        return len(self.df_time_series.index.unique())

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:


        # get_item_start = time.time()
        # time series data
        df_sample_time_series: pd.DataFrame = self.df_time_series.loc[idx]

        # Refactor: Workaround to remove skipped samples (which are too short)
        if isinstance(df_sample_time_series, pd.Series):
            df_sample_time_series = df_sample_time_series.to_frame().T

        # df_sample_time_series = df_sample_time_series.dropna(axis=1, how="all") # workarounmd related to the all  null columns from lab tests
        # logger.debug(f"{df_sample = }")

        df_sample_features_time_series = df_sample_time_series[self.features_wo_aux]
        df_sample_labels = df_sample_time_series[self.target_shifted]

        features_sample_time_series = torch.tensor(
            df_sample_features_time_series.to_numpy(dtype=np.float32)
        ).float()

        # we don't need the label for every time step (don't delete for reference)
        # target_sample = torch.tensor(df_sample[self.target_shifted].values).float()

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

        # logger.debug(f"{df_sample_static = }")
        # logger.debug(f"{features_static = }")

        target_sample_value = int(df_sample_labels.iloc[-1])
        target_sample = torch.tensor(target_sample_value).float()

        # get_item_end = time.time()
        # print(f"Epoch Time: {get_item_end - get_item_start:.4f}s")


        return features_sample_time_series, features_static, target_sample


class FixedWindowPredictionTsStaticHybridDataset(Dataset):
    def __init__(
        self,
        dataframe_time_series,
        dataframe_hybrid,
        dataframe_static,
        features: list[str],
        target: str,
        input_window_length: int = 6,
        prediction_horizon: int = 6,
    ):
        self.dataframe_time_series: pd.DataFrame = dataframe_time_series
        self.dataframe_hybrid: pd.DataFrame = dataframe_hybrid
        self.dataframe_static: pd.DataFrame = dataframe_static.loc[
            :, ["patientunitstayid", "gender", "admissionweight", "ethnicity"]
        ]
        self.features: list[str] = features
        self.features_wo_indices: list[str] = [
            x
            for x in features
            if x not in ["patientunitstayid", "time_offset", "sepsis_present", "overall_row_id", ""]
        ]
        self.target: str = target
        self.target_shifted: str = f"{target}_shifted"

        logger.info(f"{features = }")
        logger.info(f"{target = }")
        logger.info(f"{dataframe_time_series.head()}")

        self.dataframe_time_series = self.dataframe_time_series.groupby(
            "patientunitstayid", group_keys=False
        )[self.dataframe_time_series.columns].apply(lambda x: shift_and_fill_prediction_target(x))

        # Todo: Refactor dev imputation
        self.dataframe_time_series[self.features_wo_indices] = self.dataframe_time_series[
            self.features_wo_indices
        ].ffill()

        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
        scaler.fit(X=self.dataframe_time_series[self.features_wo_indices])
        self.dataframe_time_series[self.features_wo_indices] = scaler.transform(
            self.dataframe_time_series[self.features_wo_indices]
        )

        # hybrid variables

        # Todo: Refactor dev cleanup
        self.dataframe_hybrid["labresult"] = self.dataframe_hybrid["labresult"].fillna(float(0))

        self.dataframe_hybrid["labcodes"] = self.dataframe_hybrid["labname"].cat.codes

        # static variables
        self.dataframe_static = self.dataframe_static.set_index("patientunitstayid")

        df_static_transformed = pd.get_dummies(
            self.dataframe_static[["gender", "ethnicity"]], dummy_na=True
        )
        # self.dataframe_static[["gender","ethnicity"]] = df_static_transformed
        self.dataframe_static = df_static_transformed.reset_index()
        # logger.info(f"{self.dataframe.info()}")
        # logger.info(f"{self.dataframe.head()}")
        # print(df_static["gender"].head())

    def __len__(self) -> int:
        # return self.X.shape[0]
        # return len(self.dataframe)
        return len(self.dataframe_time_series.index.unique())

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # time series data
        df_sample_time_series: pd.DataFrame = self.dataframe_time_series.loc[idx]
        # Todo: This is "data cleaning" just for test purposes, to get valid nn outputs
        df_sample_time_series = df_sample_time_series.replace(to_replace=np.nan, value=float(0))
        # logger.debug(f"{df_sample = }")

        df_sample_features_time_series = df_sample_time_series[self.features_wo_indices]
        df_sample_labels = df_sample_time_series[self.target_shifted]

        features_sample_time_series = torch.tensor(
            df_sample_features_time_series.to_numpy(dtype=np.float32)
        ).float()

        # we don't need the label for every time step (don't delete for reference)
        # target_sample = torch.tensor(df_sample[self.target_shifted].values).float()

        # hybrid data

        patient_id: int = df_sample_time_series.iloc[-1]["patientunitstayid"]

        # Todo: Dev indices, has to be adjusted in make pytorch
        time_frame_start: pd.Timedelta = df_sample_time_series["time_offset"].min()
        time_frame_end: pd.Timedelta = df_sample_time_series["time_offset"].max()

        df_sample_hybrid = self.dataframe_hybrid.loc[
            self.dataframe_hybrid["patientunitstayid"] == patient_id,
            # static_var_dev
        ]

        # df_sample_hybrid.loc[:,"time_offset"] = df_sample_hybrid["time_offset"].astype("Int32")
        df_sample_hybrid.loc[:, "time_offset"] = df_sample_hybrid["time_offset"].convert_dtypes()

        # Todo: To timedelta converion in build pipeline
        # df_sample_hybrid.loc[:, "time_offset"] = pd.to_timedelta(
        # df_sample_hybrid["time_offset_min"] = pd.to_timedelta(
        new_time_offset_td = pd.to_timedelta(
            df_sample_hybrid["time_offset"],
            unit="min",
        )

        df_sample_hybrid = df_sample_hybrid.assign(time_offset=new_time_offset_td)

        # Todo: This is where the resulting dataframe is empty for some reason -> hier gibts glaube ich keine Daten, deswegen ist das Leer
        df_sample_hybrid_features = df_sample_hybrid.loc[
            (df_sample_hybrid["time_offset"] >= time_frame_start)
            & (df_sample_hybrid["time_offset"] <= time_frame_end)
        ].drop(columns="patientunitstayid")

        # Todo: Might need to refactor that out, this is pretty dense here
        df_features_hybrid_num = (
            df_sample_hybrid_features.loc[:, ["time_offset", "labresult"]]
            .groupby("time_offset", group_keys=False)["labresult"]
            # .apply(lambda x: torch.tensor(x.to_numpy(dtype=np.float32)))
            .apply(lambda x: x.to_numpy(dtype=np.float32))
        ).reset_index()

        df_features_hybrid_cat = (
            df_sample_hybrid_features.loc[:, ["time_offset", "labcodes"]]
            .groupby("time_offset", group_keys=False)["labcodes"]
            # .apply(lambda x: torch.tensor(x.to_numpy(dtype=np.int32)))
            .apply(lambda x: x.to_numpy(dtype=np.int32))
        ).reset_index()

        sequence_time_min: pd.Timedelta = df_sample_time_series["time_offset"].min()
        sequence_time_max: pd.Timedelta = df_sample_time_series["time_offset"].max()

        df_features_hybrid_num_aligned = align_to_grid(
            df_group=df_features_hybrid_num,
            pad_to_given_grid=True,
            time_grid_freq_min=5,
            time_min=sequence_time_min,
            time_max=sequence_time_max,
        )

        df_features_hybrid_cat_aligned = align_to_grid(
            df_group=df_features_hybrid_cat,
            pad_to_given_grid=True,
            time_grid_freq_min=5,
            time_min=sequence_time_min,
            time_max=sequence_time_max,
        )

        # Todo: Refactor dev replace with nan

        df_features_hybrid_num_aligned["labresult"] = df_features_hybrid_num_aligned[
            "labresult"
        ].replace(to_replace=np.nan, value=float(0))
        # df_features_hybrid_num_aligned["labresult"] = df_features_hybrid_num_aligned["labresult"].map(
        # df_features_hybrid_num_aligned["labresult"] = df_features_hybrid_num_aligned["labresult"].apply(

        # df_features_hybrid_num_aligned["labresult"] = df_features_hybrid_num_aligned.apply(
        # lambda x: [0] if ~x.isinstance(list) and x == float(0) else x,
        # axis=0
        # )
        #
        df_features_hybrid_cat_aligned = df_features_hybrid_cat_aligned.replace(
            to_replace=np.nan, value=float(0)
        )

        # df_features_hybrid_cat_aligned["labcodes"] = df_features_hybrid_cat_aligned["labcodes"].map(
        #     lambda x: [0] if x == float(0) else x,
        # )
        #

        df_features_hybrid_cat_aligned["labcodes"] = df_features_hybrid_cat_aligned["labcodes"].map(
            # lambda x: [0] if isinstance(x, float) and pd.isna(x) else x,
            lambda x: [0] if (isinstance(x, float) and x == float(0)) else x,
        )

        # features_hybrid_num = df_features_hybrid_num_aligned["labresult"].to_numpy()
        # features_hybrid_cat = df_features_hybrid_cat_aligned["labcodes"].to_numpy()

        # assert np.isnan(df_features_hybrid_cat_aligned["labcodes"].tolist()) is False, "check why there's still a nan value"

        for sublist in df_features_hybrid_num_aligned["labresult"].tolist():
            assert np.isnan(sublist).any() == False, "check why there's still a nan value"

        for sublist in df_features_hybrid_cat_aligned["labcodes"].tolist():
            assert np.isnan(sublist).any() == False, "check why there's still a nan value"

        # the number of labcodes is the same for the results (they come from the same row)
        number_hybrid_items_per_time_stamp: list[int] = [
            len(sublist) for sublist in df_features_hybrid_cat_aligned["labcodes"].tolist()
        ]
        max_hybrid_number_items_per_time_stamp: int = max(number_hybrid_items_per_time_stamp)
        number_hybrid_time_stamps = len(df_features_hybrid_cat_aligned["labcodes"])

        features_hybrid_blank: np.ndarray = np.zeros(
            (number_hybrid_time_stamps, max_hybrid_number_items_per_time_stamp)
        )
        features_hybrid_num = np.copy(features_hybrid_blank)
        features_hybrid_cat = np.copy(features_hybrid_blank)

        # Todo: Auf aligned df anpassen
        for idx, sublist in enumerate(df_features_hybrid_num_aligned["labresult"].tolist()):
            features_hybrid_num[idx, 0 : number_hybrid_items_per_time_stamp[idx]] = sublist

        for idx, sublist in enumerate(df_features_hybrid_cat_aligned["labcodes"].tolist()):
            features_hybrid_cat[idx, 0 : number_hybrid_items_per_time_stamp[idx]] = sublist

        features_hybrid_num_torch: Tensor = torch.tensor(features_hybrid_num)
        features_hybrid_cat_torch: Tensor = torch.tensor(features_hybrid_cat)

        # static data

        # we need the patient id to find the static variables for that patient
        patient_id = df_sample_time_series.iloc[-1]["patientunitstayid"]

        df_sample_static = self.dataframe_static.loc[
            self.dataframe_static["patientunitstayid"] == patient_id,
            # static_var_dev
        ]

        # Todo: Proper imputation on dataset value or somewhere else
        if len(df_sample_static) == 0:
            features_static = torch.zeros([1, len(df_sample_static.columns)]).float()
        else:
            features_static = torch.tensor(df_sample_static.to_numpy(dtype=np.float32)).float()

        features_static = features_static.squeeze()

        # logger.debug(f"{df_sample_static = }")
        # logger.debug(f"{features_static = }")

        target_sample_value = int(df_sample_labels.iloc[-1])
        target_sample = torch.tensor(target_sample_value).float()

        return (
            features_sample_time_series,
            features_hybrid_num_torch,
            features_hybrid_cat_torch,
            features_static,
            target_sample,
        )


# if __name__ == "__main__":

# notebook_path = Path.cwd()
# project_root = notebook_path.parents[0]
# dataset_train_path: Path = project_root / Path(
#     "datasets/sepsis_type_b_cohort/indexed/2024-06-24/df_train_indexed.parquet"
# )
# dataset_test_path: Path = project_root / Path(
#     "datasets/sepsis_type_b_cohort/indexed/2024-06-24/df_test_indexed.parquet"
# )
#
# dataset_static_path: Path = project_root / Path(
#     "datasets/sepsis_type_b_cohort/static/2024-06-23/dataset_static.parquet"
# )
#
# df_train = load_parquet_table(dataset_train_path)
# df_test = load_parquet_table(dataset_test_path)
#
# df_static = load_parquet_table(dataset_static_path)
#
# target = "sepsis_present"
#
# features = list(df_train.columns.difference([target]))
#
# hybrid_var_path = Path("../datasets/sepsis_type_b_cohort/hybrid/lab.parquet")
#
# df_hybrid = load_parquet_table(hybrid_var_path, backend="numpy_nullable")
# df_hybrid_test = df_hybrid.loc[:, ["patientunitstayid", "time_offset", "labresult", "labname"]]
#
# num_cats: int = len(df_hybrid_test["labname"].cat.categories)
# embedding_size = (num_cats, min(50, num_cats + 1 // 2))
# # df_train = df_dataset_cont.loc[df_dataset_cont["patientunitstayid"].isin(patient_ids[0:20])]
# # df_test = df_dataset_cont.loc[df_dataset_cont["patientunitstayid"].isin(patient_ids[21:29])]
#
# # logger.debug(f"{len(df_test) / len(df_dataset_cont) = }")
#
# # train_dataset = FixedWindowPredictionTsStaticDataset(df_train, features, target)
# train_dataset = FixedWindowPredictionTsStaticHybridDataset(
#     dataframe_time_series=df_train,
#     dataframe_hybrid=df_hybrid_test,
#     dataframe_static=df_static,
#     features=features,
#     target=target,
# )
# # test_dataset = FixedWindowPredictionTsStaticDataset(df_test, features, target)
#
# # features_sample, target_sample = train_dataset[200]
# (
#     features_sample_ts,
#     features_sample_hybrid,
#     features_sample_hybrid_num,
#     features_sample_static,
#     target_sample,
# ) = train_dataset[0]
#
# # features_sample_test, target_sample_test = test_dataset[2]
#
# logger.debug(f"{features_sample_ts.shape = }")
# logger.debug(f"{features_sample_static.shape = }")
# logger.debug(f"{target_sample.shape = }")
