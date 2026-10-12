import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from loguru import logger

from machine_intelligence.utilities.data_handling import (
    load_parquet_table,
    save_parquet_table,
    save_csv_table,
)
from machine_intelligence.utilities.dataset_setup import (
    generate_stratified_ts_datasets,
    scale_ts_variables,
    align_individual_df,
    add_sample_indices_to_df,
    align_and_add_sample_index_individual,
    dump_low_density_patient_stays,
    remove_ongoing_sepsis_sample,
    shift_target_and_drop_last_sample,
    resample_to_sepsis_prevalence,
    reindex_samplex_indices,
    determine_dataset_stats_stats,
)

from tqdm import tqdm

tqdm.pandas()


@hydra.main(
    version_base=None, config_path="../../configs", config_name="make_pytorch_dataset"
)
def main(cfg: DictConfig) -> None:
    hydra_config: dict = hydra.core.hydra_config.HydraConfig.get()
    hydra_cwd: str = hydra_config["runtime"]["cwd"]
    hydra_output_dir: str = hydra_config["runtime"]["output_dir"]

    logger.debug(f"Hydra working directory: {hydra_cwd}")
    logger.debug(f"Hydra output directory: {hydra_output_dir}")
    logger.debug(f"Loaded following configuration:\n{OmegaConf.to_yaml(cfg)}")

    use_polars: bool = cfg.use_polars
    debug_mode: bool = cfg.data.debug
    save_data: bool = cfg.data.save_on_data_generation

    logger.debug(f"{debug_mode = }")
    logger.debug(f"{use_polars = }")

    data_dir: Path = Path(cfg.data.directories.cohort)
    raw_data_path: Path = Path(cfg.data.paths.unsplitted)
    # static_data_path: Path = Path(cfg.data.paths.static_raw)

    dataset_ts_path: Path = data_dir / raw_data_path
    # dataset_static_path: Path = data_dir / static_data_path

    logger.info(f"{dataset_ts_path = }")
    # logger.info(f"{dataset_static_path = }")

    date: str = cfg.data.metadata.timestamp

    aligned_all_save_path: Path = (
            data_dir / Path(cfg.data.paths.aligned_all) / Path(date)
    )
    wo_low_density_save_path: Path = (
            data_dir / Path(cfg.data.paths.wo_low_density) / Path(date)
    )
    splitted_save_path: Path = data_dir / Path(cfg.data.paths.splitted) / Path(date)
    scaled_save_path: Path = data_dir / Path(cfg.data.paths.scaled) / Path(date)
    indexed_save_path: Path = data_dir / Path(cfg.data.paths.indexed) / Path(date)
    # static_save_path: Path = data_dir / Path(cfg.data.paths.static_processed) / Path(date)

    dataset_name_postfix: str = cfg.dataset_name_postfix

    logger.info(f"{dataset_name_postfix = }")

    # if debug_mode := False:
    if debug_mode:
        df_dataset_ts = load_parquet_table(
            parquet_path=dataset_ts_path, verbose=False, backend="numpy_nullable"
        ).loc[
                        :,
                        [
                            "patientunitstayid",
                            "time_offset",
                            "heartrate",
                            "sepsis_patient",
                            "sepsis_present",
                        ],
                        ]

        df_dataset_ts["sepsis_patient"] = df_dataset_ts["sepsis_patient"].astype(
            "category"
        )
        df_dataset_ts["sepsis_present"] = df_dataset_ts["sepsis_present"].astype(
            "category"
        )

    else:
        df_dataset_ts = load_parquet_table(
            parquet_path=dataset_ts_path, verbose=False, backend="numpy_nullable"
        )

    logger.debug(df_dataset_ts.head())
    logger.debug(df_dataset_ts.info())

    ser_empty_columns = (
        df_dataset_ts.isnull().all().loc[df_dataset_ts.isnull().all() == True]
    )
    empty_columns = ser_empty_columns.index.to_list()

    logger.debug(f"{empty_columns = }, \n dropping them from dataset")

    df_dataset_ts = df_dataset_ts.drop(columns=empty_columns)

    feature_subset: list[str] = cfg.data.considered_feature_subset_generation

    all_columns: list = df_dataset_ts.columns.tolist()
    aux_columns: list = OmegaConf.to_object(cfg.data.auxillary_columns)
    ignored_features_columns: list = OmegaConf.to_object(cfg.data.ignored_columns)
    target_columns: str = cfg.data.target_column

    considered_features = list(set(all_columns) - {target_columns} - set(aux_columns))

    input_and_prediction_window_length_h: int = cfg.data.input_and_prediction_window_h
    sample_index_column = cfg.data.sample_idx_column

    logger.info(f"{all_columns = }")
    logger.info(f"{aux_columns = }")
    logger.info(f"{ignored_features_columns = }")
    logger.info(f"{considered_features = }")
    logger.info(f"{target_columns = }")
    logger.info(f"{sample_index_column = }")

    if feature_subset is None:
        features = None
    else:
        features = feature_subset

    if debug_mode:
        filename_train_strat = (
                "df_train_stratified_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_val_strat = (
                "df_val_stratified_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_test_strat = (
                "df_test_stratified_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )

        filename_train_scaled = (
                "df_train_scaled_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_val_scaled = (
                "df_val_scaled_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_test_scaled = (
                "df_test_scaled_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )

        filename_train_idx = (
                "df_train_indexed_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_val_idx = (
                "df_val_indexed_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_test_idx = (
                "df_test_indexed_debug"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
    else:
        filename_train_strat = (
                "df_train_stratified"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_val_strat = (
                "df_val_stratified"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_test_strat = (
                "df_test_stratified"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )

        filename_train_scaled = (
                "df_train_scaled"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_val_scaled = (
                "df_val_scaled"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_test_scaled = (
                "df_test_scaled"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )

        filename_train_idx = (
                "df_train_indexed"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_val_idx = (
                "df_val_indexed"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )
        filename_test_idx = (
                "df_test_indexed"
                + f"_{input_and_prediction_window_length_h}_h"
                + dataset_name_postfix
        )

    aligned_all_save_path.mkdir(parents=True, exist_ok=True)
    wo_low_density_save_path.mkdir(parents=True, exist_ok=True)
    splitted_save_path.mkdir(parents=True, exist_ok=True)
    scaled_save_path.mkdir(parents=True, exist_ok=True)
    indexed_save_path.mkdir(parents=True, exist_ok=True)

    # df_complete_aligned = align_and_add_sample_index_individual_polars(df_split_in=df_dataset_ts)
    df_complete_aligned = align_individual_df(df=df_dataset_ts)

    total_num_sepsis_patients = df_complete_aligned.drop_duplicates(
        "patientunitstayid"
    )["sepsis_patient"].value_counts()

    logger.warning(
        f"Num sepsis patients before dropping low density stays:  {total_num_sepsis_patients = }"
    )

    save_parquet_table(
        df_in=df_complete_aligned,
        save_path=aligned_all_save_path,
        filename="df_all_aligned",
    )

    # this drops a single stray row, that might be caused by the back and forth conversion between polars and pandas
    df_complete_aligned = df_complete_aligned.dropna(subset="patientunitstayid")

    df_wo_low_density = dump_low_density_patient_stays(
        df=df_complete_aligned, density_threshold=0.75
    )

    total_num_sepsis_patients_density = df_wo_low_density.drop_duplicates(
        "patientunitstayid"
    )["sepsis_patient"].value_counts()

    logger.warning(
        f"Num sepsis patients after dropping low density stays: {total_num_sepsis_patients_density = }"
    )

    save_parquet_table(
        df_in=df_wo_low_density,
        save_path=wo_low_density_save_path,
        filename="df_wo_low_density",
    )

    df_wo_low_density_preindexed = add_sample_indices_to_df(
        df=df_wo_low_density, input_pred_window_h=input_and_prediction_window_length_h, stage="preindex"
    )

    df_train_strat, df_val_strat, df_test_strat = generate_stratified_ts_datasets(
        # df_dataset=df_wo_low_density,
        df_dataset=df_wo_low_density_preindexed,
        features=features,
        target_col="sepsis_patient",
        debug=debug_mode,
        save_sets=save_data,
    )

    total_num_sepsis_patients_train_strat = df_train_strat.drop_duplicates(
        "patientunitstayid"
    )["sepsis_patient"].value_counts()

    logger.warning(
        f"Num sepsis patients in train stratified: {total_num_sepsis_patients_train_strat= }"
    )

    # determine_dataset_stats_stats(df_train_strat, df_val_strat, df_test_strat, sample_idx_col="patientunitstayid",
    determine_dataset_stats_stats(df_train_strat, df_val_strat, df_test_strat,
                                  sample_idx_col=sample_index_column,
                                  target_col="sepsis_patient", stage="strat")

    save_parquet_table(
        df_in=df_train_strat,
        save_path=splitted_save_path,
        filename=filename_train_strat,
    )

    save_parquet_table(
        df_in=df_val_strat, save_path=splitted_save_path, filename=filename_val_strat
    )

    save_parquet_table(
        df_in=df_test_strat, save_path=splitted_save_path, filename=filename_test_strat
    )

    df_train_scaled, df_val_scaled, df_test_scaled = scale_ts_variables(
        df_train=df_train_strat,
        df_val=df_val_strat,
        df_test=df_test_strat,
        features_to_scale=considered_features,
    )

    # determine_dataset_stats_stats(df_train_strat, df_val_strat, df_test_strat, sample_idx_col="patientunitstayid",
    determine_dataset_stats_stats(df_train_strat, df_val_strat, df_test_strat, sample_idx_col=sample_index_column,
                                  target_col="sepsis_patient", stage="scaling")

    # saving the summary stats after scaling for documentation
    save_csv_table(
        df_in=df_train_scaled.describe(),
        save_path=scaled_save_path,
        filename=f"stats_{filename_train_scaled}",
    )

    save_csv_table(
        df_in=df_val_scaled.describe(),
        save_path=scaled_save_path,
        filename=f"stats_{filename_val_scaled}",
    )

    save_csv_table(
        df_in=df_test_scaled.describe(),
        save_path=scaled_save_path,
        filename=f"stats_{filename_test_scaled}",
    )

    if use_polars:
        # df_train_indexed = align_and_add_sample_index_individual_polars(df_split_in=df_train_scaled)
        # df_train_preindexed = add_sample_indices_to_df(
        #     df=df_train_scaled, input_pred_window_h=input_and_prediction_window_length_h, stage="reindex"
        # )

        df_train_preindexed = reindex_samplex_indices(df_train_scaled, sample_index_column=sample_index_column)
    else:
        df_train_indexed = align_and_add_sample_index_individual(
            df_split_in=df_train_scaled
        )

    if use_polars:
        # df_val_indexed = align_and_add_sample_index_individual_polars(df_split_in=df_val_scaled)
        # df_val_preindexed = add_sample_indices_to_df(
        #     df=df_val_scaled, input_pred_window_h=input_and_prediction_window_length_h, stage="reindex"
        # )

        df_val_preindexed = reindex_samplex_indices(df_val_scaled, sample_index_column=sample_index_column)
    else:
        df_val_indexed = align_and_add_sample_index_individual(
            df_split_in=df_val_scaled
        )
        #

    if use_polars:
        # df_test_indexed = align_and_add_sample_index_individual_polars(df_split_in=df_test_scaled)
        # df_test_preindexed = add_sample_indices_to_df(
        #     df=df_test_scaled, input_pred_window_h=input_and_prediction_window_length_h, stage="reindex"
        # )

        df_test_preindexed = reindex_samplex_indices(df_test_scaled, sample_index_column=sample_index_column)
    else:
        df_test_preindexed = align_and_add_sample_index_individual(
            df_split_in=df_test_scaled
        )

    determine_dataset_stats_stats(df_train_preindexed, df_val_preindexed, df_test_preindexed,
                                  sample_idx_col=sample_index_column,
                                  target_col="sepsis_patient", stage="preindexed")

    df_train_wo_ongoing = remove_ongoing_sepsis_sample(
        df_preindexed=df_train_preindexed, sample_index_column=sample_index_column
    )

    df_val_wo_ongoing = remove_ongoing_sepsis_sample(
        df_preindexed=df_val_preindexed, sample_index_column=sample_index_column
    )

    df_test_wo_ongoing = remove_ongoing_sepsis_sample(
        df_preindexed=df_test_preindexed, sample_index_column=sample_index_column
    )

    determine_dataset_stats_stats(df_train_wo_ongoing, df_val_wo_ongoing, df_test_wo_ongoing,
                                  sample_idx_col=sample_index_column,
                                  target_col="sepsis_sample", stage="w/o ongoing")

    df_train_shifted = shift_target_and_drop_last_sample(
        df=df_train_wo_ongoing,
        prediction_horizon_h=input_and_prediction_window_length_h,
    )

    df_val_shifted = shift_target_and_drop_last_sample(
        df=df_val_wo_ongoing, prediction_horizon_h=input_and_prediction_window_length_h
    )

    df_test_shifted = shift_target_and_drop_last_sample(
        df=df_test_wo_ongoing, prediction_horizon_h=input_and_prediction_window_length_h
    )

    determine_dataset_stats_stats(
        df_train_shifted,
        df_val_shifted,
        df_test_shifted,
        sample_idx_col=sample_index_column,
        target_col="sepsis_sample_shifted",
        stage="shifting",
    )

    total_num_sepsis_patients_train_shifted = df_train_shifted.drop_duplicates(
        "patientunitstayid"
    )["sepsis_patient"].value_counts()

    logger.warning(
        f"Num sepsis patients in train shifted: {total_num_sepsis_patients_train_shifted = }"
    )

    group_count = df_train_shifted.groupby(by=sample_index_column).count()

    logger.info(f"Every group should have the same length ! {group_count.describe()}")

    # Todo: https://imbalanced-learn.org/stable/under_sampling.html
    # Todo: Dataframe Random Under Sampler

    # we will assume that the true prevalance of sepsis is 10 %
    assumed_sepsis_prevalence_perc = cfg.data.assumed_sepsis_prevalence

    df_train_resampled = resample_to_sepsis_prevalence(
        df_train_shifted,
        sepsis_prevalence_ratio=assumed_sepsis_prevalence_perc,
        sample_index_column=sample_index_column,
    )

    df_val_resampled = resample_to_sepsis_prevalence(
        df_val_shifted,
        sepsis_prevalence_ratio=assumed_sepsis_prevalence_perc,
        sample_index_column=sample_index_column,
    )

    df_test_resampled = resample_to_sepsis_prevalence(
        df_test_shifted,
        sepsis_prevalence_ratio=assumed_sepsis_prevalence_perc,
        sample_index_column=sample_index_column,
    )

    df_train_indexed = reindex_samplex_indices(
        df=df_train_resampled, sample_index_column=sample_index_column
    )

    df_val_indexed = reindex_samplex_indices(
        df=df_val_resampled, sample_index_column=sample_index_column
    )

    df_test_indexed = reindex_samplex_indices(
        df=df_test_resampled, sample_index_column=sample_index_column
    )

    (
        train_sepsis_label_stats_pre_rus,
        val_sepsis_label_stats_pre_rus,
        test_sepsis_label_stats_pre_rus,
    ) = determine_dataset_stats_stats(
        df_train_shifted,
        df_val_shifted,
        df_test_shifted,
        sample_idx_col=sample_index_column,
        target_col="sepsis_sample_shifted",
        stage="pre_rus",
    )

    train_sepsis_label_stats_pre_rus.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_train_idx}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_pre_rus.csv",
        index=True,
    )
    val_sepsis_label_stats_pre_rus.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_val_idx}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_pre_rus.csv",
        index=True,
    )
    test_sepsis_label_stats_pre_rus.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_test_idx}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_pre_rus.csv",
        index=True,
    )

    (
        train_sepsis_label_stats_final,
        val_sepsis_label_stats_final,
        test_sepsis_label_stats_final,
    ) = determine_dataset_stats_stats(
        df_train_indexed,
        df_val_indexed,
        df_test_indexed,
        sample_idx_col=sample_index_column,
        target_col="sepsis_sample_shifted",
        stage="final",
    )

    train_sepsis_label_stats_final.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_train_idx}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_final.csv",
        index=True,
    )
    val_sepsis_label_stats_final.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_val_idx}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_final.csv",
        index=True,
    )
    test_sepsis_label_stats_final.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_test_idx}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_final.csv",
        index=True,
    )

    save_parquet_table(
        df_in=df_train_indexed,
        save_path=indexed_save_path,
        filename=filename_train_idx,
        retain_index=True,
    )

    save_parquet_table(
        df_in=df_val_indexed,
        save_path=indexed_save_path,
        filename=filename_val_idx,
        retain_index=True,
    )

    save_parquet_table(
        df_in=df_test_indexed,
        save_path=indexed_save_path,
        filename=filename_test_idx,
        retain_index=True,
    )


if __name__ == "__main__":
    main()
