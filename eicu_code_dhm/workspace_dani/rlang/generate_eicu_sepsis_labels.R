# instructions on how to use ricu https://eth-mds.github.io/ricu/reference/index.html
# the most up to date version of rico 0.6.0 has to be installed via
# remotes::install_github("eth-mds/ricu", dependencies = TRUE)
# older version suffer from profund rounding/truncation errors
# see https://github.com/eth-mds/ricu/issues/25

library(ricu)
library(units)
library(arrow)
# you may need to import some other packages as well depending on the environment

Sys.setenv(
  # the data path has to contain a folder with the name of the dataset according to ricu!
  RICU_DATA_PATH = "C:\\Users\\Daniel\\Desktop\\Alles\\Universitaet\\Master\\Thesis\\Datasets",
  RICU_SRC_LOAD = "eicu"
)

ricu::setup_src_data("eicu")
ricu::import_src("eicu")
ricu::attach_src("eicu")
ricu::src_data_avail()

# Note:
# The base class of the interval argument, e.g. mins or hours
# sets the type of the offsets, to said base class. The numbers are being rounded
# to the "precision" specified by e.g. mins(15L).
# This means, that the time unit is minutes and that the timesteps are rounded to 15min precision.

# Additional Note: The 15 min precision is due to the fact that R is unable to use "long vectors" in data.table.
# This results in a error with respect to the SOFA-concept -> which in turn limits the precision of the sepsis labels
# since "sep3" excepts both inputs to be the same length

# parameters from multicenter-sepsis https://github.com/BorgwardtLab/multicenter-sepsis
# sofa_data <- ricu::load_concepts("sofa",
#                                  "eicu",
#                                  keep_components = TRUE,
#                                  #interval = mins(15L)
#                                  interval = hours(1L)
# )
#
# si_data <- ricu::load_concepts("susp_inf", "eicu",
#                                abx_min_count = 2L,
#                                positive_cultures = TRUE,
#                                si_mode = "or",
#                                keep_components = TRUE,
#                                #interval = mins(15L)
#                                interval = hours(1L)
# )
#
# sepsis_data <- sep3(sofa_data, si_data, si_window = "any",
#                     #si_lwr = mins(2880L), # 60 * 48 = 2880
#                     #si_upr = mins(1440L), # 60 * 24 = 1440
#                     si_lwr = hours(48L), # 60 * 48 = 2880
#                     si_upr = hours(24L), # 60 * 24 = 1440
#                     keep_components = TRUE,
#                     #interval = mins(15L)
#                     #interval = hours(0.25L)
#                     interval = hours(1L)
# )
#

si <- load_concepts("susp_inf", "eicu", abx_min_count = 2L, positive_cultures = TRUE, si_mode = "or", keep_components = TRUE,
                    #interval = mins(15L)
                    interval = hours(1L)
)

pids <- c(unique(id_col(si))[1:200], 141436)

sofa <- load_concepts("sofa", "eicu", keep_components = TRUE,
                      interval = hours(1L),
                      #interval = mins(15L),
                      patient_ids = pids)

sep3(sofa[patientunitstayid == 141436], si[patientunitstayid == 141436],
     si_window = "any",
     #     si_lwr = mins(2880L), # 60 * 48 = 2880
     #     si_upr = mins(1440L), # 60 * 24 = 1440
     keep_components = TRUE,
     #    interval = mins(15L)
     interval = hours(1L)
)


# uncomment if you want to save the data to a parquet file!
#arrow::write_parquet(sepsis_data, "./sepsis_data_plecko_workaround.parquet")
#arrow::write_parquet(sofa_data, "./sofa_data_plecko_workaround.parquet")
#arrow::write_parquet(si_data, "./si_data_plecko_workaround.parquet")