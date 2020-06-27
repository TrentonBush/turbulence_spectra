import pandas as pd
from scipy.io import loadmat
import numpy as np
from pathlib import Path


def matlab_to_pandas(filepath: Path) -> pd.DataFrame:
    exclusions = {
        "__header__",
        "__version__",
        "__globals__",
        "tower",
        "datastream",
    }
    matlab = loadmat(filepath, squeeze_me=True)

    sample_freq = matlab["tower"]["daqfreq"].item()
    if sample_freq != 20:
        raise ValueError(f"Unexpected sample_freq: {sample_freq}")

    utc_offset = matlab["tower"]["UTCoffset"].item()
    if utc_offset != -7:
        raise ValueError(f"Unexpected utc_offset: {utc_offset}")

    df = pd.DataFrame()
    for key in matlab.keys():
        if key not in exclusions:
            df[key] = matlab[key]["val"].item()

    return df


def sonic_summary(df: pd.DataFrame, height: str):
    raise NotImplementedError  # need spectral density functions
    out = {}
    horizontal = f"Sonic_CupEqHorizSpeed_{height}m"
    vertical = f"Sonic_z_clean_{height}m"
    direction = f"Sonic_direction_{height}m"

    out["nan_count"] = df[horizontal].isna().sum()
    out["mean"] = df[horizontal].mean()
    out["mean_square"] = df[horizontal].pow(2).mean()
    out["mean_cube"] = df[horizontal].pow(3).mean()

    out["nan_count_vert"] = df[vertical].isna().sum()
    out["mean_vert"] = df[vertical].mean()
    out["mean_square_vert"] = df[vertical].pow(2).mean()
    out["mean_cube_vert"] = df[vertical].pow(3).mean()

    out["mean_dir"] = df[direction].mean()
    out["waked_frac"] = (
        df[[direction]].query(f"80 < {direction} < 210").count()[0] / 12000
    )

    return out
