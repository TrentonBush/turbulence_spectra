# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, Dict
import pandas as pd
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


def summarize_file(filepath: Path,) -> Dict[str, pd.DataFrame]:
    timestamp = pd.to_datetime(
        filepath.name[:16], format="%m_%d_%Y_%H_%M"
    ) - pd.Timedelta(-7, unit="h")
    matlab_df = pmat.matlab_to_pandas(filepath)
    sonics = []
    cups = []

    for height in SONIC_HEIGHTS:
        sonics.append(pd.DataFrame(pmat.sonic_summary(matlab_df, height), index=[0]))
    for cup in CUP_NAMES:
        cups.append(pd.DataFrame(pmat.cup_summary(matlab_df, cup), index=[0]))

    sonics = pd.concat(sonics, ignore_index=True)
    sonics["timestamp"] = timestamp
    cups = pd.concat(cups, ignore_index=True)
    cups["timestamp"] = timestamp

    misc = pd.DataFrame(pmat.misc_summary(matlab_df), index=[0])
    misc["timestamp"] = timestamp

    return {"sonics": sonics, "cups": cups, "misc": misc}


def main(source_dir: Union[Path, str], out_dir: Union[Path, str]) -> None:
    input_files = list(Path(source_dir).glob("*.mat"))

    # initialize - convert dict of dfs to dict of lists of dfs
    out_dfs = summarize_file(input_files[0])
    out_dfs = dict(zip(out_dfs.keys(), map(lambda x: [x], out_dfs.values())))

    for file in input_files[1:]:
        dfs = summarize_file(file)
        for key in dfs.keys():
            out_dfs[key].append(dfs[key])

    for key, val in out_dfs.items():
        filename = Path(out_dir) / f"{key}.parquet"
        pd.concat(val, ignore_index=True).to_parquet(filename, index=False)


if __name__ == "__main__":
    start = pd.Timestamp("now")
    source_dir = Path("./data/raw/")
    assert source_dir.exists()
    out_dir = Path("./data/processed/")
    assert out_dir.exists()
    main(source_dir, out_dir)
    print(f"Duration: {(pd.Timestamp('now') - start).round('1s')}")
