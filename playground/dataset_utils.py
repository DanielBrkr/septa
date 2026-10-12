import torch.nn as nn
import pandas as pd
import math
from functools import reduce
from loguru import logger
from datetime import timedelta

from tqdm import tqdm

def collate_fn_variable_length_batch_pad(batch):
    batch_inputs, batch_labels = zip(*batch)

    batch_inputs_original_length = [len(x) for x in batch_inputs]
    batch_labels_original_length = [len(x) for x in batch_labels]

    batch_inputs_padded = nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True)
    batch_labels_padded = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)

    return (
        batch_inputs_padded,
        batch_labels_padded,
        batch_inputs_original_length,
        batch_labels_original_length,
    )


def align_to_grid(df_group: pd.DataFrame.groupby, time_grid_freq: str = "5min") -> pd.DataFrame:
    time_grid_floor = df_group["time_offset"].min().floor(time_grid_freq)
    time_grid_ceiling = df_group["time_offset"].max().ceil(time_grid_freq)

    df_grid_only = pd.DataFrame(
        data={
            "time_offset": pd.timedelta_range(
                start=time_grid_floor, end=time_grid_ceiling, freq=time_grid_freq
            )
        }
    )

    df_group_gridded = pd.merge_asof(
        left=df_grid_only, right=df_group, on="time_offset", direction="forward"
    )

    return df_group_gridded


def shift_and_fill_prediction_target(
    df_group: pd.DataFrame.groupby, prediction_horizon_h: int = 4, time_grid_freq: str = "5min"
) -> pd.DataFrame:
    prediction_horizon = pd.to_timedelta(prediction_horizon_h, unit="h")
    prediction_target_shifted = "sepsis_present_shifted"

    prediction_horizon_time_steps_number = int(prediction_horizon / pd.to_timedelta(time_grid_freq))

    df_group[prediction_target_shifted] = df_group["sepsis_present"].shift(
        -prediction_horizon_time_steps_number
    )

    with pd.option_context("future.no_silent_downcasting", True):
        df_group[prediction_target_shifted] = (
            df_group["sepsis_present_shifted"].ffill().infer_objects(copy=False)
        )

    return df_group


def chunk_df_based_on_time(
    df_group: pd.DataFrame.groupby, total_sample_counter, input_window_h: int = 6
) -> pd.DataFrame:
    """This function provides chunks of a patients data while keeping track of the overall number of resulting samples.
    We need to keep track of the overall sample number to make it easier to load the proper sample with the pytorch
    dataloader, as the dataloader expects an iterable which we provide why the overall sample number index.
    Chunk refers to a fixed window slice of the dataframe"""

    chunking_sample_min: timedelta = df_group["time_offset"].min()
    chunking_sample_max: timedelta = df_group["time_offset"].max()

    chunking_sample_entire_length: timedelta = abs(chunking_sample_max - chunking_sample_min)

    input_window_length: timedelta = pd.to_timedelta(input_window_h, unit="h")

    number_of_chunks_sample: int = math.floor(chunking_sample_entire_length / input_window_length)

    list_df_individual_chunks: list = []

    for i in range(number_of_chunks_sample):
        chunk_time_lower: timedelta = (
            input_window_length * i + chunking_sample_min
        )  # offset to account for ts that don't start at 0
        chunk_time_upper: timedelta = input_window_length * (i + 1) + chunking_sample_min

        # logger.debug(f"{i = }, {chunk_time_lower = }, {chunk_time_upper = }")

        df_chunk_of_time_range: pd.DataFrame = df_group.loc[
            (df_group["time_offset"] >= chunk_time_lower)
            & (df_group["time_offset"] <= chunk_time_upper)
        ]

        df_chunk_of_time_range = df_chunk_of_time_range.copy()
        df_chunk_of_time_range["patient_sample_count"] = total_sample_counter.value()

        df_chunk_of_time_range = df_chunk_of_time_range.set_index(
            ["patient_sample_count"], inplace=False
        )

        list_df_individual_chunks.append(df_chunk_of_time_range)
        # logger.debug(f"{df_chunk_of_time_range = }")

        df_group_tracked: pd.DataFrame = reduce(
            lambda left, right: pd.concat([left, right]), list_df_individual_chunks
        )

        total_sample_counter.increment()

        # logger.debug(f"{total_sample_counter.value() = }")

    if number_of_chunks_sample == 0:
        offending_id = df_group["patientunitstayid"].iloc[-1]
        logger.error(
            f"Sample to short, skipping! {number_of_chunks_sample = } for {offending_id = }"
        )

    else:
        return df_group_tracked


def add_overall_sample_idx_to_df(df_for_dataloader: pd.DataFrame) -> pd.DataFrame:
    class _SampleCount:
        """Helper class that tracks the total number of samples for overall multiindex"""

        def __init__(self):
            self.total_sample_count: int = 0

        def increment(self):
            self.total_sample_count += 1

        def value(self):
            return self.total_sample_count

    overall_sample_counter = _SampleCount()

    tqdm.pandas(desc="Adding overall sample idx!")

    df_for_dataloader_indexed: pd.DataFrame = df_for_dataloader.groupby(
        by="patientunitstayid", group_keys=False
    )[df_for_dataloader.columns].progress_apply(
        lambda x: chunk_df_based_on_time(df_group=x, total_sample_counter=overall_sample_counter)
    )

    return df_for_dataloader_indexed
