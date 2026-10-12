#!/bin/bash

# Note: You need to change the paths in the configurations accordingly still for steps that build upon each other,
# they have to be consistent with the input and output data that you expect !

set -e # bash exits if any command fails
DEBUG_MODE=False
COHORT_TYPE="A" # Med-Surg ICU
#COHORT_TYPE="B" # Cardiac ICU

echo "Building new dataset! Starting with cohorted data! Debug=${DEBUG_MODE}, Cohort Type=${COHORT_TYPE}"


if [ $COHORT_TYPE == "A" ]; then

  python workspace_dani/data_preprocessing/merge_tables_pandas.py data=merge cohort@data.cohort=sepsis_type_a &&

  python workspace_dani/data_preprocessing/dataset_cleanup.py data=post_merge_cleanup cohort@data.cohort=sepsis_type_a

  python workspace_dani/data_preprocessing/build_dataset.py data=build_dataset_sepsis cohort@data.cohort=sepsis_type_a

  python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_a_cohort_time_series &&

  python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_a_cohort_static

elif [ $COHORT_TYPE == "B" ]; then

  python workspace_dani/data_preprocessing/merge_tables_pandas.py data=merge cohort@data.cohort=sepsis_type_b &&

  python workspace_dani/data_preprocessing/dataset_cleanup.py data=post_merge_cleanup cohort@data.cohort=sepsis_type_b &&

  python workspace_dani/data_preprocessing/build_dataset.py data=build_dataset_sepsis cohort@data.cohort=sepsis_type_b &&

   python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_a_cohort_time_series &&

  python workspace_dani/data_preprocessing/data_quality_dataset.py data=data_quality_assessment_dataset debug_mode=False \
   draft_mode=False data.which_stage=dataset dataset@data.dataset=sepsis_type_a_cohort_static
else
  echo "Specified cohort type doesn't exist ${COHORT_TYPE}"
  exit 1
fi;

echo "New dataset done!"