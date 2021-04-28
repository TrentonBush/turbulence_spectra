"""CLI script to extract the cup anemometer subset of 1Hz data from the full 20Hz archive"""

from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
from scipy.io.matlab.miobase import MatReadError
from tqdm import tqdm
import multiprocessing
from functools import partial
import logging
import typer

import src.data.process_matlab as pmat
from src.data.make_dataset import create_logger


def load_cups_with_error_handling(
    filepath: Path, logger: logging.Logger
) -> pd.DataFrame:
    try:
        return pmat.matlab_to_pandas(
            filepath, timestamps=True, col_subset=pmat.CUP_SUBSET, drop_raw_time=True,
        )
    except (OSError, MatReadError, ValueError):
        message = f"Load error: {filepath.name}"
        tqdm.write(message + ". See log for details")
        logger.exception(message)
        return pd.DataFrame()  # empty


def concat_cups_multiprocess(
    input_files: Sequence[Path], n_processes: Optional[int] = None,
) -> pd.DataFrame:
    """extract cup, vane, and icing (temp & dew point) data from each file and write to .parquet file

    Parameters
    ----------
    input_files : Sequence[Path]
        iterable of .mat files
    n_processes : Optional[int], optional
        number of processes to use, by default None, which is interpreted as os.cpu_count()

    Returns
    -------
    [pd.DataFrame]
        concatenated results from each input file
    """
    logger = create_logger()
    count = len(input_files)
    func = partial(load_cups_with_error_handling, logger=logger)

    logger.info(f"Processing {count} files")
    tqdm.write(f"Processing {count} files in parallel")
    start = pd.Timestamp("now")

    with multiprocessing.Pool(processes=n_processes) as pool:
        processed_files = list(
            tqdm(pool.imap(func, input_files, chunksize=10), total=count,)
        )
    combined = pd.concat(processed_files)
    end = pd.Timestamp("now")
    duration = (end - start).round("s")
    message = f"{count} files processed in {duration}. Seconds per file: {(duration / count).total_seconds():.3f}"

    logger.info(message)
    tqdm.write(message)

    return combined


def main(source_dir: str, out_dir: str) -> None:
    """extract data from .mat files in source_dir and write results to out_dir/cups_1hz.parquet

    Parameters
    ----------
    source_dir : str
        directory containing *.mat files
    out_dir : str
        directory to write resulting .parquet file
    """
    logger = create_logger(dest=Path(out_dir) / "extract_cups_1hz.log")
    logger.info("Starting")
    input_files = list(Path(source_dir).glob("*.mat"))
    combined = concat_cups_multiprocess(input_files)

    filename = Path(out_dir) / f"cups_1hz.parquet"
    combined.to_parquet(filename, index=False)


typer.run(main)
