# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
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


def summarize_file(filepath: Path,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamp = pd.to_datetime(
        filepath.name[:16], format="%m_%d_%Y_%H_%M"
    ) + pd.Timedelta(pmat.EXPECTED_UTC_OFFSET, unit="h")
    matlab_df = pmat.matlab_to_pandas(filepath)
    sonics = []
    cups = []

    for height in SONIC_HEIGHTS:
        sonics.append(pd.DataFrame(pmat.sonic_summary(matlab_df, height), index=[0]))
    for cup in CUP_NAMES:
        cups.append(pd.DataFrame(pmat.cup_summary(matlab_df, cup), index=[0]))

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
    count = len(input_files)
    logging.info(f"Processing {count} files")
    print(f"Processing {count} files")
    start = pd.Timestamp("now")
    with Pool(processes=n_processes) as pool:
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
    message = "\n".join(
        [
            f"Files: {count}",
            f"Elapsed time: {duration}",
            f"Seconds per file: {(duration / count).total_seconds():.3f}",
        ]
    )
    logging.info(message)
    print(message)
    return out_dict


def main(source_dir: str, out_dir: str) -> None:
    logging.basicConfig(
        filename=Path(out_dir) / "process.log",
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )
    input_files = list(Path(source_dir).glob("*.mat"))
    out_dfs = summarize_many_multiprocess(input_files)

    for key, val in out_dfs.items():
        filename = Path(out_dir) / f"{key}.parquet"
        val.to_parquet(filename, index=False)


if __name__ == "__main__":
    typer.run(main)
