import pandas as pd
from loguru import logger
from pathlib import Path
import pyarrow as pa
from typing import Literal


def load_parquet_table(
    parquet_path: Path,
    engine: Literal["pyarrow", "fastparquet"] = "pyarrow",
    backend: Literal["pyarrow", "numpy_nullable"] = "pyarrow",
    debug_mode: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a parquet file into a pandas DataFrame.

    Note: debug_mode

    This function attempts to load a parquet file specified by the `parquet_path` parameter into a pandas DataFrame
    using the pyarrow engine.
    It logs the attempt, the summary of the loaded DataFrame,
    and the data types of the DataFrame columns.


    Returns:
    - pd.DataFrame or None: A pandas DataFrame containing the data from the parquet file if successful; otherwise,
     None if the file does not exist or an error occurs during loading.

    Note:
    - This function uses the 'pyarrow' engine by default for loading the parquet file.

    Example usage:
    >>> df = load_parquet_table(Path('/path/to/your/file.parquet'))
    >>> print(df)
    """

    logger.info(f"Trying to load from {parquet_path = }")

    if parquet_path.is_file():
        if debug_mode:
            try:
                if engine == "pyarrow":
                    df_table = pd.read_parquet(
                        parquet_path,
                        engine=engine,
                        dtype_backend=backend,
                    )

                    df_table = df_table.head(n=10000)

                else:
                    df_table = pd.read_parquet(
                        parquet_path,
                        engine=engine,
                    )
                    df_table = df_table.head(n=10000)

            except ValueError as ex:
                logger.warning(
                    f"Probably facing a pandas pyarrow parse error! Ignoring metadata. \n Error: {ex}"
                )
                df_table = pa.parquet.read_table(source=parquet_path).to_pandas(
                    ignore_metadata=True
                )

        else:
            try:
                if engine == "pyarrow":
                    df_table = pd.read_parquet(
                        parquet_path,
                        engine=engine,
                        dtype_backend=backend,
                    )
                else:
                    df_table = pd.read_parquet(
                        parquet_path,
                        engine=engine,
                    )

            except ValueError as ex:
                logger.warning(
                    f"Probably facing a pandas pyarrow parse error! Ignoring metadata. \n Error: {ex}"
                )
                df_table = pa.parquet.read_table(source=parquet_path).to_pandas(
                    ignore_metadata=True
                )

        if verbose:
            print(f"{'-' * 100}")
            logger.info("Loading! Dataframe summary:")
            logger.info(f"{df_table.info(verbose=False)}")

    else:
        logger.error("Parquet File not found")
        df_table = None

    return df_table


def save_parquet_table(
    df_in: pd.DataFrame, save_path: Path, filename: str, retain_index: bool = False
):
    save_path.mkdir(parents=True, exist_ok=True)
    parquet_save_path = save_path / Path(filename + ".parquet")

    print(f"{'-' * 100}")
    logger.info("Saving! Dataframe summary:")
    logger.info(f"{df_in.info(verbose=False)}")

    df_in.to_parquet(path=parquet_save_path, index=retain_index)

    logger.info(f"Saved dataframe to {parquet_save_path}")
    logger.success(f"Dataframe {filename} has been saved as a parquet file.")

def save_csv_table(
        df_in: pd.DataFrame, save_path: Path, filename: str
):
    save_path.mkdir(parents=True, exist_ok=True)
    csv_save_path = save_path / Path(filename + ".csv")

    print(f"{'-' * 100}")
    logger.info("Saving! Dataframe summary:")
    logger.info(f"{df_in.info(verbose=False)}")

    df_in.to_csv(path_or_buf=csv_save_path, decimal=".")

    logger.info(f"Saved dataframe to {csv_save_path}")
    logger.success(f"Dataframe {filename} has been saved as a csv file.")
