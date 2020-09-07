# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict
import pandas as pd
from scipy.io.matlab.miobase import MatReadError
from tqdm import tqdm
import multiprocessing
import logging
import typer

import src.data.process_matlab as pmat

SONIC_HEIGHTS = (119, 100, 74, 61, 41, 15)
CUP_NAMES = (
    "Cup_WS_C1_130m",
    "Cup_WS_122m",
    "Cup_WS_C1_105m",
    "Cup_WS_87m",
    "Cup_WS_C1_80m",
    "Cup_WS_C1_55m",
    "Cup_WS_38m",
    "Cup_WS_C1_30m",
    "Cup_WS_10m",
    "Cup_WS_3m",
)


def create_logger():
    """helper func to set up loggers within multiprocessing functions

    Returns
    -------
    logging.Logger
        logger
    """
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s| %(levelname)s| %(processName)s] %(message)s"
    )
    handler = logging.FileHandler(Path("./data/processed/process.log"))
    handler.setFormatter(formatter)

    # prevents duplicated messages
    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger


def summarize_file(filepath: Path,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """extract features from a 10-minute matlab file

    Parameters
    ----------
    filepath : Path
        .mat filepath

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        resulting dataframes, (sonics, cups, miscellaneous)
    """
    logger = create_logger()
    timestamp = pd.to_datetime(
        filepath.name[:16], format="%m_%d_%Y_%H_%M"
    ) + pd.Timedelta(pmat.EXPECTED_UTC_OFFSET, unit="h")
    try:
        matlab_df = pmat.matlab_to_pandas(filepath)
    except (OSError, MatReadError, ValueError) as e:
        message = f"Skipped {filepath.name} due to load error"
        tqdm.write(message)
        logger.exception(message)
        logger.exception(e)
        return tuple([pd.DataFrame()] * 3)  # empty
    sonics = []
    cups = []
    try:
        for height in SONIC_HEIGHTS:
            sonics.append(
                pd.DataFrame(pmat.sonic_summary(matlab_df, height), index=[0])
            )
        for cup in CUP_NAMES:
            cups.append(pd.DataFrame(pmat.cup_summary(matlab_df, cup), index=[0]))
    except ValueError as e:
        message = f"Skipped {filepath.name} due to processing error"
        tqdm.write(message)
        logger.exception(message)
        logger.exception(e)
        return tuple([pd.DataFrame()] * 3)  # empty

    sonics = pd.concat(sonics, ignore_index=True)
    sonics["timestamp"] = timestamp  # type: ignore
    cups = pd.concat(cups, ignore_index=True)
    cups["timestamp"] = timestamp  # type: ignore

    misc = pd.DataFrame(pmat.misc_summary(matlab_df), index=[0])
    misc["timestamp"] = timestamp

    return (sonics, cups, misc)


def summarize_many_multiprocess(
    input_files: Sequence[Path], n_processes: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """apply summarize_file to many files using multiprocessing

    Parameters
    ----------
    input_files : Sequence[Path]
        iterable of .mat files
    n_processes : Optional[int], optional
        number of processes to use, by default None, which is interpreted as os.cpu_count()

    Returns
    -------
    Dict[str, pd.DataFrame]
        results for each instrument type, concatenated across all input files
        Keys: "sonic", "cup", "misc"
    """
    logger = create_logger()
    count = len(input_files)
    logger.info(f"Processing {count} files")
    tqdm.write(f"Processing {count} files")
    start = pd.Timestamp("now")
    with multiprocessing.Pool(processes=n_processes) as pool:
        processed_files = list(
            tqdm(
                pool.imap(summarize_file, input_files, chunksize=10),
                total=len(input_files),
            )
        )
    combined = map(pd.concat, zip(*processed_files))
    out_dict = dict(zip(["sonic", "cup", "misc"], combined))
    end = pd.Timestamp("now")
    duration = (end - start).round("s")
    message = f"{count} files processed in {duration}. Seconds per file: {(duration / count).total_seconds():.3f}"
    logger.info(message)
    tqdm.write(message)
    return out_dict


def main(source_dir: str, out_dir: str) -> None:
    """summarize all .mat files in source_dir and write results as .parquet files to out_dir

    Parameters
    ----------
    source_dir : str
        directory containing *.mat files
    out_dir : str
        directory to write resulting .parquet files
    """
    input_files = list(Path(source_dir).glob("*.mat"))
    out_dfs = summarize_many_multiprocess(input_files)

    for key, val in out_dfs.items():
        filename = Path(out_dir) / f"{key}.parquet"
        val.to_parquet(filename, index=False)


if __name__ == "__main__":
    logger = create_logger()
    logger.info("Starting")
    typer.run(main)
