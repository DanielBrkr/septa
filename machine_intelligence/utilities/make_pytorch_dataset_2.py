import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from loguru import logger

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from machine_intelligence.utilities.data_handling import (
    load_parquet_table,
    save_parquet_table,

    save_csv_table,
)
from machine_intelligence.utilities.dataset_setup import (

    align_individual_df,
    add_sample_indices_to_df,
    remove_ongoing_sepsis_sample,
    shift_target_and_drop_last_sample,
    scale_ts_variables,
    determine_dataset_stats_stats,
    resample_to_sepsis_prevalence,
    reindex_samplex_indices,
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

    df_dataset_ts["patientunitstayid"] = df_dataset_ts["patientunitstayid"].astype(int)

    # df_complete_aligned = align_and_add_sample_index_individual_polars(df_split_in=df_dataset_ts)
    df_complete_aligned = align_individual_df(df=df_dataset_ts)

    total_num_sepsis_patients = df_complete_aligned.drop_duplicates(
        "patientunitstayid"
    )["sepsis_patient"].value_counts()

    logger.warning(
        f"Num sepsis patients aligned:  {total_num_sepsis_patients = }"
    )

    save_parquet_table(
        df_in=df_complete_aligned,
        save_path=aligned_all_save_path,
        filename="df_all_aligned",
    )

    # this drops a single stray row, that might be caused by the back and forth conversion between polars and pandas
    df_complete_aligned = df_complete_aligned.dropna(subset="patientunitstayid")

    density_threshold = cfg.data.density_threshold

    df_wo_low_density = df_complete_aligned.groupby(by="patientunitstayid")[df_complete_aligned.columns].apply(
        lambda x: x if (x["heartrate"].isna().sum() / len(x["heartrate"])) < density_threshold else None).reset_index(
        drop=True)

    df_wo_low_density_preindexed = add_sample_indices_to_df(
        df=df_wo_low_density, input_pred_window_h=input_and_prediction_window_length_h, stage="preindex"
        # df=df_complete_aligned, input_pred_window_h=input_and_prediction_window_length_h, stage="preindex"
    )

    sepsis_sample_stats = df_wo_low_density_preindexed.drop_duplicates(
        sample_index_column
    )["sepsis_patient"].value_counts()

    logger.warning(
        f"indexed: {sepsis_sample_stats = }"
    )

    df_wo_low_density_wo_ongoing = remove_ongoing_sepsis_sample(
        df_preindexed=df_wo_low_density_preindexed, sample_index_column=sample_index_column
    )

    sepsis_sample_stats = df_wo_low_density_wo_ongoing.drop_duplicates(
        sample_index_column
    )["sepsis_sample"].value_counts()

    logger.warning(
        f"w/ ongoing: {sepsis_sample_stats = }"
    )

    df_shifted = shift_target_and_drop_last_sample(
        df=df_wo_low_density_wo_ongoing,
        prediction_horizon_h=input_and_prediction_window_length_h,
    )

    sepsis_sample_stats = df_shifted.drop_duplicates(
        sample_index_column
    )["sepsis_sample_shifted"].value_counts()

    logger.warning(
        f"Shifted: {sepsis_sample_stats = }"
    )

    density_threshold_negative_sample = density_threshold

    def discard_low_density_negative_samples(df_group: pd.DataFrame.groupby):
        if (df_group["heartrate"].isna().sum() / len(
                df_group["heartrate"])) > density_threshold_negative_sample and not (
                df_group["sepsis_sample_shifted"].any()):
            df_group = None
        else:
            df_group = df_group

        return df_group

    # Todo: Verify that this works properly
    df_shifted_wo_low_density = df_shifted.groupby(by=sample_index_column)[df_shifted.columns].apply(
        lambda x: discard_low_density_negative_samples(x)
    ).reset_index(drop=True)

    sepsis_sample_stats = df_shifted_wo_low_density.drop_duplicates(
        sample_index_column
    )["sepsis_sample_shifted"].value_counts()

    logger.warning(
        f"Shifted wo low density: {sepsis_sample_stats = }"
    )

    # saving the resulting dataset w/o low density and ongoing samples

    save_parquet_table(
        df_in=df_shifted_wo_low_density ,
        save_path=wo_low_density_save_path,
        # filename=filename_train_idx,
        filename=f"df_all_wo_ongoing_wo_low_dens_{density_threshold}_sep_prev_only_test_{resample_only_test}_{assumed_sepsis_prevalence_perc}_imp_{imputation_strategy}",
        retain_index=True,
    )


    ### median imputation

    imputation_strategy = cfg.data.imputation_strategy

    if imputation_strategy == "none":
        df_imputed = df_shifted_wo_low_density
        logger.warning(f"{imputation_strategy = }")

    if imputation_strategy == "median":
        imputer = SimpleImputer(strategy="median")
        columns_to_impute = df_shifted_wo_low_density.columns[df_shifted_wo_low_density.columns != "time_offset"]

        df_imputed_values = imputer.fit_transform(
            df_shifted_wo_low_density.loc[:, columns_to_impute])
        df_imputed = pd.DataFrame(df_imputed_values, columns=columns_to_impute, index=df_shifted_wo_low_density.index)

        df_imputed["time_offset"] = df_shifted_wo_low_density["time_offset"]

        logger.warning(f"{imputation_strategy = }")

    df_stratification = (
        # df_shifted.loc[:, [sample_index_column, "sepsis_sample_shifted"]].copy().drop_duplicates()
        # df_shifted_wo_low_density.loc[:, [sample_index_column, "sepsis_sample_shifted"]].copy().drop_duplicates()
        df_shifted_wo_low_density.loc[:, [sample_index_column, "sepsis_sample_shifted"]].copy().drop_duplicates()
    )
    df_stratification["sepsis_sample_shifted"] = df_stratification[["sepsis_sample_shifted"]].astype("boolean")
    df_stratification.head()

    if debug_strat := False:
        patient_ids_all = df_stratification["patientunitstayid"].unique().tolist()
        patient_ids_debug = patient_ids_all[::50]
        df_stratification = df_stratification.loc[
            df_stratification["patientunitstayid"].isin(patient_ids_debug)
        ]

    x_initial_split = df_stratification[sample_index_column].reset_index(drop=True)
    y_initial_split = df_stratification["sepsis_sample_shifted"].reset_index(drop=True)

    sss_train_test = StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, test_size=0.2, random_state=42
    )
    sss_train_val = StratifiedShuffleSplit(
        n_splits=1, train_size=0.75, test_size=0.25, random_state=42
    )

    #  this should appease the lsp
    train_val_indices = None
    test_indices = None
    train_indices = None
    val_indices = None

    for i, (train_val, test) in enumerate(sss_train_test.split(x_initial_split, y_initial_split)):
        train_val_indices = train_val
        test_indices = test

    x_train_val = x_initial_split.loc[train_val_indices].reset_index(drop=True)
    x_test = x_initial_split.loc[test_indices].reset_index(drop=True)

    y_train_val = y_initial_split.loc[train_val_indices].reset_index(drop=True)
    y_test = y_initial_split.loc[test_indices].reset_index(drop=True)

    for i, (train, val) in enumerate(sss_train_val.split(x_train_val, y_train_val)):
        train_indices = train
        val_indices = val

    x_train = x_train_val.loc[train_indices]
    x_val = x_train_val.loc[val_indices]

    # we end up with train, val, test ratios of 0.6, 0.2, 0.2
    total_samples = len(x_initial_split)

    logger.info(
        f"Total samples: {total_samples}, Train, Validation, Test - Ratios: "
        f"{len(x_train) / total_samples}, {len(x_val) / total_samples} , {len(x_test) / total_samples}"
    )

    # Todo: Part of the verification
    # df_train_strat = df_shifted.loc[
    #                  df_shifted[sample_index_column].isin([*x_train]), :
    #                  ]
    # df_val_strat = df_shifted.loc[
    #                df_shifted[sample_index_column].isin([*x_val]), :
    #                ]
    # df_test_strat = df_shifted.loc[
    #                 df_shifted[sample_index_column].isin([*x_test]), :
    #                 ]

    df_train_strat = df_imputed.loc[
                     df_imputed[sample_index_column].isin([*x_train]), :
                     ]
    df_val_strat = df_imputed.loc[
                   df_imputed[sample_index_column].isin([*x_val]), :
                   ]
    df_test_strat = df_imputed.loc[
                    df_imputed[sample_index_column].isin([*x_test]), :
                    ]

    df_train_strat["sepsis_sample_shifted"] = df_train_strat[["sepsis_sample_shifted"]].astype("boolean")
    df_val_strat["sepsis_sample_shifted"] = df_val_strat[["sepsis_sample_shifted"]].astype("boolean")
    df_test_strat["sepsis_sample_shifted"] = df_test_strat[["sepsis_sample_shifted"]].astype("boolean")

    train_label_stats = df_train_strat.drop_duplicates(subset=sample_index_column)[
        "sepsis_sample_shifted"
    ].value_counts()
    val_label_stats = df_val_strat.drop_duplicates(subset=sample_index_column)[
        "sepsis_sample_shifted"
    ].value_counts()
    test_label_stats = df_test_strat.drop_duplicates(subset=sample_index_column)[
        "sepsis_sample_shifted"
    ].value_counts()

    train_label_ratio = train_label_stats.loc[True] / train_label_stats.loc[False]
    val_label_ratio = val_label_stats.loc[True] / val_label_stats.loc[False]
    test_label_ratio = test_label_stats.loc[True] / test_label_stats.loc[False]

    logger.info(
        f"Train Dataset Sepsis (False vs. True) Ratio: {train_label_ratio:.2%}, #Sepsis Labels: {train_label_stats.loc[True]}"
    )
    logger.info(
        f"Valid Dataset Sepsis (False vs. True) Ratio: {val_label_ratio:.2%}, #Sepsis Labels: {val_label_stats.loc[True]}"
    )
    logger.info(
        f"Test Dataset Sepsis (False vs. True) Ratio: {test_label_ratio:.2%}, #Sepsis Labels: {test_label_stats.loc[True]}"
    )

    df_train_scaled, df_val_scaled, df_test_scaled = scale_ts_variables(
        df_train=df_train_strat,
        df_val=df_val_strat,
        df_test=df_test_strat,
        features_to_scale=considered_features,
        method="MinMax"
    )

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

    (
        train_sepsis_label_stats_pre_rus,
        val_sepsis_label_stats_pre_rus,
        test_sepsis_label_stats_pre_rus,
    ) = determine_dataset_stats_stats(
        df_train_scaled,
        df_val_scaled,
        df_test_scaled,
        sample_idx_col=sample_index_column,
        target_col="sepsis_sample_shifted",
        stage="pre_rus", )

    # we will assume that the true prevalance of sepsis is 10 %
    assumed_sepsis_prevalence_perc = cfg.data.assumed_sepsis_prevalence

    train_sepsis_label_stats_pre_rus.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_train_idx}_dens_{density_threshold}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_pre_rus.csv",
        index=True,
    )
    val_sepsis_label_stats_pre_rus.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_val_idx}_dens_{density_threshold}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_pre_rus.csv",
        index=True,
    )
    test_sepsis_label_stats_pre_rus.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_test_idx}_dens_{density_threshold}_sep_prev_{assumed_sepsis_prevalence_perc}_label_stats_pre_rus.csv",
        index=True,
    )

    if resample_only_test := cfg.data.resample_only_test:
        df_train_resampled = df_train_scaled

        df_val_resampled = df_val_scaled

        df_test_resampled = resample_to_sepsis_prevalence(
            df_test_scaled,
            sepsis_prevalence_ratio=assumed_sepsis_prevalence_perc,
            sample_index_column=sample_index_column,
        )

    else:

        df_train_resampled = resample_to_sepsis_prevalence(
            df_train_scaled,
            sepsis_prevalence_ratio=assumed_sepsis_prevalence_perc,
            sample_index_column=sample_index_column,
        )

        df_val_resampled = resample_to_sepsis_prevalence(
            df_val_scaled,
            sepsis_prevalence_ratio=assumed_sepsis_prevalence_perc,
            sample_index_column=sample_index_column,
        )

        df_test_resampled = resample_to_sepsis_prevalence(
            df_test_scaled,
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
                    / f"{filename_train_idx}_dens_{density_threshold}_sep_prev_only_test_{resample_only_test}_{assumed_sepsis_prevalence_perc}_imp_{imputation_strategy}_label_stats_final.csv",
        index=True,
    )
    val_sepsis_label_stats_final.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_val_idx}_dens_{density_threshold}_sep_prev_only_test_{resample_only_test}_{assumed_sepsis_prevalence_perc}_imp_{imputation_strategy}_label_stats_final.csv",
        index=True,
    )
    test_sepsis_label_stats_final.to_csv(
        path_or_buf=indexed_save_path
                    / f"{filename_test_idx}_dens_{density_threshold}_sep_prev_only_test_{resample_only_test}_{assumed_sepsis_prevalence_perc}_imp_{imputation_strategy}_label_stats_final.csv",
        index=True,
    )

    save_parquet_table(
        df_in=df_train_indexed,
        save_path=indexed_save_path,
        # filename=filename_train_idx,
        filename=f"{filename_train_idx}_dens_{density_threshold}_sep_prev_only_test_{resample_only_test}_{assumed_sepsis_prevalence_perc}_imp_{imputation_strategy}",
        retain_index=True,
    )

    save_parquet_table(
        df_in=df_val_indexed,
        save_path=indexed_save_path,
        # filename=filename_val_idx,
        filename=f"{filename_val_idx}_dens_{density_threshold}_sep_prev_only_test_{resample_only_test}_{assumed_sepsis_prevalence_perc}_imp_{imputation_strategy}",
        retain_index=True,
    )

    save_parquet_table(
        df_in=df_test_indexed,
        save_path=indexed_save_path,
        # filename=filename_test_idx,
        filename=f"{filename_test_idx}_dens_{density_threshold}_sep_prev_only_test_{resample_only_test}_{assumed_sepsis_prevalence_perc}_imp_{imputation_strategy}",
        retain_index=True,
    )


if __name__ == "__main__":
    main()
