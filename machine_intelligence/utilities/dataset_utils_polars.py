from __future__ import annotations

import pandas as pd
import polars as pl
from datetime import timedelta, datetime
import math
from functools import reduce
from tqdm import tqdm
from loguru import logger


class SampleCount:
    """Helper class that tracks the total number of samples for overall multiindex"""

    def __init__(self):
        self.total_sample_count: int = 0

    def increment(self):
        self.total_sample_count += 1

    def value(self):
        return self.total_sample_count

    def reset(self):
        self.total_sample_count = 0


def chunk_df_based_on_time_polars(
        df_group: pd.DataFrame, total_sample_counter, stage: str, input_window_h: int = 6
) -> pl.DataFrame:
    """This function provides chunks of a patients data while keeping track of the overall number of resulting samples.
    We need to keep track of the overall sample number to make it easier to load the proper sample with the pytorch
    dataloader, as the dataloader expects an iterable which we provide why the overall sample number index.
    Chunk refers to a fixed window slice of the dataframe"""

    pl.Config(tbl_rows=-1)

    # polars can't handle pandas category columns, thats a workaround
    for column in df_group.select_dtypes(include=["category"]).columns:
        df_group[column] = df_group[column].astype(str)

    df_group = pl.from_pandas(df_group)

    chunking_sample_min: timedelta = df_group["time_offset"].min()
    chunking_sample_max: timedelta = df_group["time_offset"].max()

    chunking_sample_entire_length: timedelta = abs(chunking_sample_max - chunking_sample_min)

    input_window_length: timedelta = timedelta(hours=input_window_h)
    sample_idx_suffix: str = f"{input_window_h}_h"
    sample_idx_name: str = f"sample_idx_{sample_idx_suffix}"

    number_of_chunks_sample: int = math.floor(chunking_sample_entire_length / input_window_length)

    list_df_individual_chunks: list = []

    if number_of_chunks_sample == 0:
        offending_id = df_group["patientunitstayid"].to_list()[-1]
        print(
            f"Sample too short, skipping! number_of_chunks_sample = {number_of_chunks_sample} for offending_id = {offending_id}. {total_sample_counter.value() = }"
        )
        if stage == "reindex":
            print("debug")

        return None

    else:
        for i in range(number_of_chunks_sample):
            chunk_time_lower: timedelta = (
                    input_window_length * i + chunking_sample_min
            )  # offset to account for ts that don't start at 0
            chunk_time_upper: timedelta = input_window_length * (i + 1) + chunking_sample_min

            df_chunk_of_time_range: pl.DataFrame = df_group.filter(
                # (df_group["time_offset"] >= chunk_time_lower)
                (df_group["time_offset"] >= chunk_time_lower)  # Todo: Determine effect
                & (df_group["time_offset"] < chunk_time_upper)
            )

            df_chunk_of_time_range = df_chunk_of_time_range.with_columns(
                pl.lit(total_sample_counter.value()).alias(sample_idx_name)
            )

            # Todo: Workaround for samples that don't have the expected number of samples
            #  it seems like they are not properly aligned since the datapoints are too far apart

            expected_number_time_steps = input_window_length / timedelta(minutes=5)
            actual_number_time_steps = len(df_chunk_of_time_range)

            if actual_number_time_steps == expected_number_time_steps:
                list_df_individual_chunks.append(df_chunk_of_time_range)
                total_sample_counter.increment()

            else:
                logger.warning(
                    f"Chunk not complete! {actual_number_time_steps} out of {expected_number_time_steps}, at {total_sample_counter.value() = }"
                )


        if not list_df_individual_chunks:
            return None

        df_group_tracked = reduce(lambda left, right: left.vstack(right), list_df_individual_chunks)

        # df_group_tracked.group_by(pl.col(sample_idx_name)).agg(
        #     pl.col("time_offset").max() - pl.col("time_offset").min()
        #     # > pl.duration(hours=input_window_h) # Todo: Ist das evtl der grund?
        #      > pl.duration(hours=input_window_h)
        # )

        df_group_tracked_violations = df_group_tracked.group_by(pl.col(sample_idx_name)).agg(
            (pl.col("time_offset").max() - pl.col("time_offset").min()).alias("invalid")
            > pl.duration(hours=input_window_h)
        )


        # Todo: Refactor dev workaround, we are dropping the entire dataframe right know when a corrupted samples is being encountered
        if df_group_tracked_violations.select(pl.col("invalid").any()).item() is True:
            logger.warning("Sample with too long time span encountered, dropping sample !")
            return None
        else:
            df_group_tracked = df_group_tracked.to_pandas()
            return df_group_tracked


def add_overall_sample_idx_to_df_polars(df_for_dataloader: pd.DataFrame, input_pred_window_h: int,
                                        stage: str) -> pd.DataFrame:
    overall_sample_counter = SampleCount()

    tqdm.pandas(desc="Adding overall sample idx")

    logger.info("Starting to add overall sample index for pytorch dataloader!")

    df_for_dataloader_indexed: pd.DataFrame = df_for_dataloader.groupby(
        by="patientunitstayid", group_keys=False
    )[df_for_dataloader.columns].progress_apply(
        lambda x: chunk_df_based_on_time_polars(
            df_group=x, total_sample_counter=overall_sample_counter, stage=stage, input_window_h=input_pred_window_h
        )
    )

    return df_for_dataloader_indexed


def align_to_grid_polars(
        df_group: pd.DataFrame.groupby | pd.DataFrame,
        time_grid_freq_min: int = 5,
        pad_to_given_grid: bool = False,
        time_min: pd.Timedelta | None = None,
        time_max: pd.Timedelta | None = None,
) -> pd.DataFrame:
    time_grid_freq = f"{time_grid_freq_min}min"

    assert pad_to_given_grid == True & isinstance(
        time_min, pd.Timedelta
    ), "You have to specify the min and max bounds of the time sequence"

    if pad_to_given_grid:
        time_grid_floor = time_min.floor(time_grid_freq)
        time_grid_ceiling = time_max.ceil(time_grid_freq)
    else:
        time_grid_floor = df_group["time_offset"].min().floor(time_grid_freq)
        time_grid_ceiling = df_group["time_offset"].max().ceil(time_grid_freq)

    df_grid_only = pd.DataFrame(
        data={
            "time_offset": pd.timedelta_range(
                start=time_grid_floor, end=time_grid_ceiling, freq=time_grid_freq
            )
        }
    )

    for column in df_group.select_dtypes(include=["category"]).columns:
        df_group[column] = df_group[column].astype(str)

    # df_group = pl.from_pandas(df_group)
    df_grid_only = pl.from_pandas(df_grid_only)

    df_group_gridded = df_grid_only.join_asof(
        df_group,
        on="time_offset",
        strategy="backward",
        tolerance=pd.to_timedelta(time_grid_freq_min - 1, unit="min"),
        # this prevents from making up data inbetween time step holes, the last value will be assigned to the next time steps
    )

    df_group_gridded = df_group_gridded.to_pandas()

    # df_group.select(pl.col("time_offset") + datetime(2024, 1, 1))

    return df_group_gridded


def align_to_grid_polars_pure(
        df_group: pl.DataFrame.group_by | pl.DataFrame,
        time_grid_freq_min: int = 5,
        pad_to_given_grid: bool = False,
        time_min: pd.Timedelta | None = None,
        time_max: pd.Timedelta | None = None,
) -> pd.DataFrame:
    pl.Config(tbl_rows=-1)

    time_grid_freq = f"{time_grid_freq_min}min"

    assert pad_to_given_grid == True & isinstance(
        time_min, pd.Timedelta
    ), "You have to specify the min and max bounds of the time sequence"

    if pad_to_given_grid:
        time_grid_start = time_min.floor(time_grid_freq)
        time_grid_end = time_max.ceil(time_grid_freq)
    else:
        time_min = df_group.select(pl.col("time_offset")).min().item()
        time_max = df_group.select(pl.col("time_offset")).max().item()

        # time_grid_floor = pd.to_timedelta(time_min).floor(time_grid_freq)
        time_grid_start = pd.to_timedelta(time_min).ceil(time_grid_freq)
        time_grid_end = pd.to_timedelta(time_max).ceil(time_grid_freq)

    df_grid_only = pd.DataFrame(
        data={
            "time_offset": pd.timedelta_range(
                start=time_grid_start, end=time_grid_end, freq=time_grid_freq
            )
        }
    )

    df_grid_only = pl.from_pandas(df_grid_only)

    # this a workaround for polars, group by dynamic doesn't work with a duration
    dt_workaround_offset = datetime(2024, 1, 1)
    df_group_date = df_group.with_columns(pl.col("time_offset") + dt_workaround_offset)

    df_grid_only_date = df_grid_only.with_columns(pl.col("time_offset") + dt_workaround_offset)

    # more or less a workaround, such that we can keep all the data during a grid tick,
    # preserves especially sparse data -> median of a single number is the number itself
    df_group_aggregated = df_group_date.group_by_dynamic(
        index_column="time_offset",
        every="5m",
        closed="left",
        start_by="window",
        include_boundaries=True,
        # ).agg(pl.all().median())
    ).agg(pl.all().last())

    df_group_gridded_agg = df_grid_only_date.join_asof(
        df_group_aggregated,
        left_on="time_offset",
        right_on="_upper_boundary",
        strategy="backward",
        tolerance=pd.to_timedelta(time_grid_freq_min, unit="min"),
        # this prevents from making up data inbetween time step holes, the last value will be assigned to the next time steps
        coalesce=True,
    )

    df_group_gridded_agg = df_group_gridded_agg.drop(
        ["_lower_boundary", "_upper_boundary", "time_offset_right"]
    )
    df_group_gridded_agg = df_group_gridded_agg.with_columns(
        pl.col("time_offset") - dt_workaround_offset
    )

    # we have to add the patient id and labels manually when there's absolutely no data at a given grid tick
    # rows start to go missing otherwise, since we need the patient id at any given point
    df_group_gridded_agg = df_group_gridded_agg.with_columns(
        pl.col("patientunitstayid").forward_fill(),
        pl.col("sepsis_patient").forward_fill(),
        pl.col("sepsis_present").forward_fill(),

    )

    # if len(df_group.select(pl.col("sepsis_present").value_counts()).unnest("sepsis_present")) > 1:
    #     logger.warning(f"Sepsis sample!")

    # debug_id = 427103
    # if (
    #     len(
    #         df_group_gridded_agg.unique(subset="patientunitstayid").filter(
    #             pl.col("patientunitstayid") == debug_id
    #         )
    #     )
    #     == 1
    # ):
    #     logger.warning(f"Debug id: {debug_id}")

    return df_group_gridded_agg
