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
  # which is "eicu" in our case
  RICU_DATA_PATH = "C:\\Users\\Daniel\\Desktop\\Alles\\Universitaet\\Master\\Thesis\\Datasets",
  RICU_SRC_LOAD = "eicu"
)

ricu::import_src("eicu")
ricu::attach_src("eicu")
ricu::src_data_avail()

# Note:
# The base class of the interval argument, e.g. mins or hours
# sets the type of the offsets, to said base class. The numbers are being rounded
# to the "precision" specified by e.g. mins(15L).
# This means, that the time unit is minutes and that the timesteps are rounded to 15min precision.

# Additional Note: The 15 min precision is due to the fact that R is unable to use "long vectors" in data.table
# This results in a error with respect to the SOFA-concept -> which in turn limits the calculation of the sepsis labels
# since "sep3" excepts both inputs to be the same length

sofa_data <- ricu::load_concepts("sofa",
                                 "eicu",
                                 keep_components = FALSE,
                                 interval = hours(1L),
#                                 interval = mins(15L
#                                  interval = mins(5L)
#                                  interval = hours(0.083L), # 1 / 60 * 5
                                 patient_ids = c(935416L, 3220164L)


)

# uncomment if you want to save the data to a parquet file!
arrow::write_parquet(sofa_data, "./sofa_data_5min.parquet")
