#!/bin/bash

# Note: You need to change the paths in the configurations accordingly still for steps that build upon each other,
# they have to be consistent with the input and output data that you expect !

set -e # bash exits if any command fails
DEBUG_MODE=False

USE_AUTOVIZ=False
USE_YDPROFILING=False

echo "Determining Data Quality ! AutoViz=${USE_AUTOVIZ}, YDPROFILING= ${USE_YDPROFILING}, Debug=${DEBUG_MODE}, Cohort Type=${COHORT_TYPE}"

python workspace_dani/data_preprocessing/data_quality_dataset.py --multirun data=data_quality_assessment_dataset \
  dataset@data.dataset=sepsis_type_b_cohort_time_series,sepsis_type_b_cohort_static,sepsis_type_a_cohort_time_series,sepsis_type_a_cohort_static \
  debug_mode=False draft_mode=False data.which_stage=dataset


echo "Data Quality analysis done!"