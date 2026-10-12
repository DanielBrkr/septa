import pandas as pd
from tqdm.auto import tqdm
from loguru import logger

tqdm.pandas()


def attach_sepsis_labels_to_merged_tables(
    df_merged_tabs: pd.DataFrame, df_sepsis_labels: pd.DataFrame
) -> pd.DataFrame:
    def add_sepsis_label_per_patient(patient_group: pd.DataFrame.groupby):
        patient_id = patient_group["patientunitstayid"].iloc[0]

        df_patient_group_to_label = patient_group.copy()
        df_patient_group_to_label["sepsis_patient"] = pd.Series(dtype=bool)
        df_patient_group_to_label["sepsis_present"] = pd.Series(dtype=bool)
        df_patient_group_to_label["time_offset"] = pd.to_timedelta(
            df_patient_group_to_label["time_offset"], unit="min"
        )

        if patient_id in patient_ids_with_sepsis:
            # patient has a  sepsis label
            sepsis_label = df_sepsis_labels[
                df_sepsis_labels["patientunitstayid"] == patient_id
            ]
            sepsis_label = sepsis_label.rename(columns={"sepsis_offset": "time_offset"})

            # Todo: This might be redundant if I set the dtype to timedelta64[s] explicitly in the dataset build pipeline
            sepsis_label["time_offset"] = sepsis_label["time_offset"].astype(
                "timedelta64[ns]"
            )

            sepsis_patient_offset = sepsis_label.loc[:, "time_offset"].values[0]

            patient_exact_sepsis_label_row = pd.DataFrame(
                data={
                    "time_offset": [sepsis_patient_offset],
                    "patientunitstayid": [patient_id],
                }
            )

            df_patient_labeled = pd.merge(
                left=df_patient_group_to_label,
                right=patient_exact_sepsis_label_row,
                on=["patientunitstayid", "time_offset"],
                how="outer",
            )

            df_patient_labeled.loc[
                df_patient_labeled["time_offset"] < sepsis_patient_offset,
                ["sepsis_present"],
            ] = False
            df_patient_labeled.loc[
                df_patient_labeled["time_offset"] >= sepsis_patient_offset,
                ["sepsis_present"],
            ] = True

            df_patient_labeled["sepsis_patient"] = True

        else:
            # patient doesn't have a sepsis label
            df_patient_group_to_label["sepsis_patient"] = False
            df_patient_labeled = df_patient_group_to_label.assign(sepsis_present=False)

        return df_patient_labeled

    patient_ids_with_sepsis = df_sepsis_labels["patientunitstayid"].unique().tolist()

    logger.info("Adding sepsis labels to merged tables..")

    df_dataset_with_sepsis = df_merged_tabs.groupby(
        by="patientunitstayid", group_keys=False
    ).progress_apply(add_sepsis_label_per_patient, include_groups=True)

    return df_dataset_with_sepsis
