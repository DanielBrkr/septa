#!/bin/bash

# Note: You need to change the paths in the configurations accordingly still for steps that build upon each other,
# they have to be consistent with the input and output data that you expect !

set -e # bash exits if any command fails
DEBUG_MODE=True
COHORT_TYPE="A" # Med-Surg ICU + Cardiac ICU
#COHORT_TYPE="B" # Cardiac ICU

echo "Merging time stamps of the lab table! Debug=${DEBUG_MODE}, Cohort Type=${COHORT_TYPE}"


if [ $COHORT_TYPE == "A" ]; then

  python workspace_dani/data_preprocessing/lab_to_proper_time_series.py cohort@data.cohort=sepsis_type_a data=lab_proper_ts_cohort_a \
   table@data.cohort.table=lab debug_mode=${DEBUG_MODE} draft_mode=False


elif [ $COHORT_TYPE == "B" ]; then

  python workspace_dani/data_preprocessing/lab_to_proper_time_series.py cohort@data.cohort=sepsis_type_b data=lab_proper_ts_cohort_b \
   table@data.cohort.table=lab debug_mode=${DEBUG_MODE} draft_mode=False

else
  echo "Specified cohort type doesn't exist ${COHORT_TYPE}"
  exit 1
fi;

echo "Consolidating lab time series done!"