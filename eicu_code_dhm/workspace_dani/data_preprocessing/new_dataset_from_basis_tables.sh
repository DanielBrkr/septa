#!/bin/bash

# Note: You need to change the paths in the configurations accordingly still for steps that build upon each other,
# they have to be consistent with the input and output data that you expect !

set -e # bash exits if any command fails
DEBUG_MODE=False
#DEBUG_MODE=True

#DATASET_TABLES="apache_aps_var,apache_groups,apache_patient_results,apache_pred_var,diagnosis,lab,patient,vital_aperiodic,vital_periodic"
DATASET_TABLES="apache_patient_results,lab,patient,vital_aperiodic,vital_periodic,apache_pred_var"
COHORT_TYPE="A" # Med-Surg ICU, MICU + Cardiac ICU
#COHORT_TYPE="B" # Cardiac ICUi

echo "Building new dataset! Starting with cleanup! Debug=${DEBUG_MODE}, Cohort Type=${COHORT_TYPE}"


python workspace_dani/data_preprocessing/drop_columns.py --multirun data=drop_columns \
table@data.table=${DATASET_TABLES} \
debug_mode=${DEBUG_MODE} &&

python workspace_dani/data_preprocessing/table_cleanup.py \
--multirun data=cleanup table@data.table=${DATASET_TABLES} \
debug_mode=${DEBUG_MODE} &&


./workspace_dani/data_preprocessing/new_cohorts_ids.sh
./workspace_dani/data_preprocessing/new_sepsis_labels_and_ids.sh


if [ $COHORT_TYPE == "A" ]; then
  python workspace_dani/data_preprocessing/reduce_to_cohort.py \
  --multirun table@cohort.table=${DATASET_TABLES} cohort=sepsis_type_a \
  debug_mode=${DEBUG_MODE} draft_mode=False &&

  python workspace_dani/data_preprocessing/lab_to_proper_time_series.py cohort@data.cohort=sepsis_type_a data=lab_proper_ts_cohort_a \
   table@data.cohort.table=lab debug_mode=${DEBUG_MODE} draft_mode=False &&

  python workspace_dani/data_preprocessing/merge_tables_pandas.py data=merge cohort@data.cohort=sepsis_type_a &&


  python workspace_dani/data_preprocessing/build_dataset.py data=build_dataset_sepsis cohort@data.cohort=sepsis_type_a &&

  python workspace_dani/data_preprocessing/dataset_cleanup.py data=post_merge_cleanup cohort@data.cohort=sepsis_type_a &&

  python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_a_cohort_time_series &&

  python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_a_cohort_static

elif [ $COHORT_TYPE == "B" ]; then

  python workspace_dani/data_preprocessing/reduce_to_cohort.py \
  --multirun table@cohort.table=${DATASET_TABLES} cohort=sepsis_type_b \
  debug_mode=${DEBUG_MODE} draft_mode=False &&
  # Note: This saves the new lab parquet file to the dir consolidated, you need to move manually to the dir cohorted and overwrite
#  python workspace_dani/data_preprocessing/lab_to_proper_time_series.py cohort@data.cohort=sepsis_type_b data=lab_proper_ts_cohort_b \
#   table@data.cohort.table=lab debug_mode=${DEBUG_MODE} draft_mode=False &&


  python workspace_dani/data_preprocessing/lab_to_proper_time_series.py cohort@data.cohort=sepsis_type_b data=lab_proper_ts_cohort_b \
   table@data.cohort.table=lab debug_mode=${DEBUG_MODE} draft_mode=False &&

  python workspace_dani/data_preprocessing/merge_tables_pandas.py data=merge cohort@data.cohort=sepsis_type_b &&

  python workspace_dani/data_preprocessing/build_dataset.py data=build_dataset_sepsis cohort@data.cohort=sepsis_type_b

  python workspace_dani/data_preprocessing/dataset_cleanup.py data=post_merge_cleanup cohort@data.cohort=sepsis_type_b &&

  python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_b_cohort_time_series &&

  python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_b_cohort_static

else
  echo "Specified cohort type doesn't exist ${COHORT_TYPE}"
  exit 1
fi;

echo "New dataset done!"