from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from data_preprocessing.utils.data_handling import load_parquet_table

"""**Attention:** This is just a prototype dataset to test the dataloader, 
don't use it w/o modifications!"""


class ExpandingTimeSeriesPanelDataset(Dataset):
    def __init__(self, dataframe, features: list[str], target: str):
        self.dataframe: pd.DataFrame = dataframe
        self.index_offset = self.dataframe.index[0]
        self.features: list[str] = features
        self.features_wo_indices: list[str] = [x for x in features if
                                               x not in ["patientunitstayid", "time_offset", "sepsis_present",
                                                         "overall_row_id"]]
        self.target: str = target

        logger.info(f"{self.index_offset = }")
        logger.info(f"{features = }")
        logger.info(f"{target = }")

        self.dataframe = self.dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:

        # initializing the actual index, the default pytorch sampler starts with 0
        actual_idx: int = idx + self.index_offset

        # don't use loc with idx without a bracket here, that returns a series instead of a df
        # df_at_index: pd.DataFrame = self.dataframe.loc[[idx], [self.features, self.target]]

        df_at_index: pd.DataFrame = self.dataframe.loc[[actual_idx]]
        # df_at_index: pd.DataFrame = self.dataframe.loc[self.dataframe["overall_row_id"] == actual_idx]#self.dataframe.loc[["overall_row_id"]]

        patient_id_at_index: int = df_at_index.loc[actual_idx, "patientunitstayid"]

        df_of_patient: pd.DataFrame = self.dataframe.loc[self.dataframe["patientunitstayid"] == patient_id_at_index]

        df_before_idx_of_patient: pd.DataFrame = df_of_patient.loc[df_of_patient.index < actual_idx]

        if df_before_idx_of_patient.empty is True:
            # First available sample x_(t-k)
            # logger.debug("df_before evaluated to empty")
            df_sample = df_at_index
        else:

            # logger.debug("df_before didn't evaluated to empty")
            # samples up to (including) x_t
            df_sample = pd.concat([df_before_idx_of_patient, df_at_index], axis=0)

        # logger.debug(df_at_index.head())
        # logger.debug(f"{patient_id_at_index = }")
        # logger.debug(f"{df_before_idx_of_patient = }")
        # logger.debug(f"{actual_idx = }")
        # logger.debug(f"{df_sample = }")

        # Todo: This is "data cleaning" just for test purposes

        df_sample = df_sample.replace(to_replace=np.nan, value=float(0))
        # logger.debug(f"{df_sample = }")

        features_sample = torch.tensor(df_sample[self.features_wo_indices].values).float()
        target_sample = torch.tensor(df_sample[self.target].values).float()

        # logger.debug(f"{features_sample.shape = }")
        # logger.debug(f"{target_sample.shape = }")

        return features_sample, target_sample


if __name__ == "__main__":
    notebook_path = Path.cwd()
    project_root = notebook_path.parents[0]
    dataset_ts_path: Path = project_root / Path(
        "datasets/raw_and_shitty/dataset_ts.parquet")
    # %%
    df_dataset_ts = load_parquet_table(parquet_path=dataset_ts_path, verbose=False)
    df_dataset_ts.head()

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

    train_dataset = TimeSeriesPanelDataset(df_train, features, target)

    features_sample, target_sample = train_dataset[5000]

    logger.debug(f"{features_sample.shape = }")
    logger.debug(f"{target_sample.shape = }")
