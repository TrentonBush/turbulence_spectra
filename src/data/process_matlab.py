import pandas as pd
from scipy.io import loadmat
import numpy as np
from pathlib import Path


def matlab_to_pandas(filepath: Path):
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

