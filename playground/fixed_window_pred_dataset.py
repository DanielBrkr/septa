from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from data_preprocessing.utils.data_handling import load_parquet_table
from playground.dataset_utils import align_to_grid, shift_and_fill_prediction_target, add_overall_sample_idx_to_df


class FixedWindowPredictionDataset(Dataset):
    def __init__(self, dataframe, features: list[str], target: str, input_window_length: int = 6,
                 prediction_horizon: int = 6):
        self.dataframe: pd.DataFrame = dataframe
        self.features: list[str] = features
        self.features_wo_indices: list[str] = [x for x in features if
                                               x not in ["patientunitstayid", "time_offset", "sepsis_present","overall_row_id", ""]]
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

        self.dataframe = self.dataframe[self.dataframe["time_offset"] >= pd.to_timedelta(0)]

        self.dataframe = self.dataframe.groupby("patientunitstayid", group_keys=False)[self.dataframe.columns].apply(
            lambda x: align_to_grid(x)).reset_index(drop=True)

        self.dataframe = self.dataframe.groupby("patientunitstayid", group_keys=False)[
        self.dataframe.columns].apply(lambda x: shift_and_fill_prediction_target(x)).reset_index(drop=True)

        self.dataframe = add_overall_sample_idx_to_df(df_for_dataloader=self.dataframe)

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

        features_sample = torch.tensor(df_sample_features.values).float()

        # we don't need the label for every time step (don't delete for reference)
        # target_sample = torch.tensor(df_sample[self.target_shifted].values).float()

        target_sample_value = int(df_sample_labels.iloc[-1])
        target_sample = torch.tensor(target_sample_value).float()

        return features_sample, target_sample


if __name__ == "__main__":
    notebook_path = Path.cwd()
    project_root = notebook_path.parents[0]
    dataset_ts_path: Path = project_root / Path(
        "datasets/raw_and_shitty/dataset_ts.parquet")

    df_dataset_ts = load_parquet_table(parquet_path=dataset_ts_path, verbose=False)

    df_dataset_ts.head()
    df_dataset_ts.info()

    patient_ids = df_dataset_ts["patientunitstayid"].unique().tolist()
    df_dataset_cont_all = df_dataset_ts.loc[:,
                          ["patientunitstayid", "time_offset", "heartrate", "sao2", "temperature", "st1",
                           "sepsis_present"]]
    df_dataset_cont = df_dataset_cont_all[df_dataset_cont_all["patientunitstayid"].isin(patient_ids[0:29])].copy()

    target = "sepsis_present"

    features = list(df_dataset_cont.columns.difference([target]))

    df_train = df_dataset_cont.loc[df_dataset_cont["patientunitstayid"].isin(patient_ids[0:20])]
    df_test = df_dataset_cont.loc[df_dataset_cont["patientunitstayid"].isin(patient_ids[21:29])]

    logger.debug(f"{len(df_test) / len(df_dataset_cont) = }")

    train_dataset = FixedWindowPredictionDataset(df_train, features, target)
    test_dataset = FixedWindowPredictionDataset(df_test, features, target)

    features_sample, target_sample = train_dataset[200]

    features_sample_test, target_sample_test = test_dataset[100]

    # logger.debug(f"{features_sample.shape = }")
    # logger.debug(f"{target_sample.shape = }")
