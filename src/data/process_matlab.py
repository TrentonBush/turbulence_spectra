"""module to extract and aggregate high frequency data to low frequency metrics"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from typing import Tuple, Optional, Sequence, Dict, Union

import src.data.aggregate as aggregate


SONIC_SAMPLE_FREQ = 20
CUP_SAMPLE_FREQ = 1
EXPECTED_UTC_OFFSET = -7
SONIC_SUBSET = (
    "Air_Temp_38m",
    "DeltaT_122_87m",
    "DeltaT_87_38m",
    "PRECIP_INTEN",
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
    "time_UTC",
    "Sonic_CupEqHorizSpeed_119m",
    "Sonic_direction_119m",
    "Sonic_z_clean_119m",
    "Sonic_CupEqHorizSpeed_100m",
    "Sonic_direction_100m",
    "Sonic_z_clean_100m",
    "Sonic_CupEqHorizSpeed_74m",
    "Sonic_direction_74m",
    "Sonic_z_clean_74m",
    "Sonic_CupEqHorizSpeed_61m",
    "Sonic_direction_61m",
    "Sonic_z_clean_61m",
    "Sonic_CupEqHorizSpeed_41m",
    "Sonic_direction_41m",
    "Sonic_z_clean_41m",
    "Sonic_CupEqHorizSpeed_15m",
    "Sonic_direction_15m",
    "Sonic_z_clean_15m",
)
CUP_SUBSET = (
    "time_UTC",
    "Air_Temp_87m",
    "DeltaT_122_87m",
    "Dewpt_Temp_122m",
    "Dewpt_Temp_87m",
    "PRECIP_INTEN",
    "Cup_WS_C1_130m",
    "Cup_WS_122m",
    "Cup_WS_C1_105m",
    "Vane_WD_122m",
    "Vane_WD_87m",
)


def matlab_to_pandas(
    filepath: Path,
    timestamps: bool = False,
    col_subset: Optional[Sequence] = SONIC_SUBSET,
    drop_raw_time: bool = True,
) -> pd.DataFrame:
    """read an NREL NWTC matlab file and load the data into a pandas dataframe

    Args:
        filepath (Path): path to file
        timestamps (bool, optional): if True, read timestamp column and set as dataframe index. Defaults to False.
        col_subset (Optional[Sequence], optional): optionally load only a subset of column names. Defaults to SONIC_SUBSET.
        drop_raw_time (bool, optional): if True, drop the raw matlab time column, which is a float indicating 'days since 0 AD'. Defaults to True.

    Raises:
        ValueError: if the sample frequency stored in the matlab file does not match the expected value SONIC_SAMPLE_FREQ
        ValueError: if the UTC offset stored in the matlab file does not match the expected value EXPECTED_UTC_OFFSET
        ValueError: if the timestamp encoded in the filename does not match the first timestamp of the actual data

    Returns:
        pd.DataFrame: dataframe of high frequency data
    """
    exclusions = {
        "__header__",
        "__version__",
        "__globals__",
        "tower",
        "datastream",
    }
    if col_subset is CUP_SUBSET:
        stride = 20  # correct for oversampling 1Hz data at 20Hz
    else:
        stride = 1
    filepath = Path(filepath)
    matlab = loadmat(filepath, squeeze_me=True)

    sample_freq = matlab["tower"]["daqfreq"].item()
    if sample_freq != SONIC_SAMPLE_FREQ:
        raise ValueError(f"Unexpected sample_freq: {sample_freq}")

    utc_offset = matlab["tower"]["UTCoffset"].item()
    if utc_offset != EXPECTED_UTC_OFFSET:
        raise ValueError(f"Unexpected utc_offset: {utc_offset}.")

    if col_subset is None:  # get all 150 cols
        col_subset = [key for key in matlab.keys() if key not in exclusions]

    data = {
        key : matlab[key]["val"].item()[::stride].astype(np.float32)
        for key in col_subset
        if key != "time_UTC"  # time needs 64 bit precision
    }
    data["time_UTC"] = matlab["time_UTC"]["val"].item()[::stride]
    df = pd.DataFrame(data)

    # Convert times from matlab to pandas
    # Matlab format is "days since 0 AD"
    # offset is days from 0 AD to start of Unix Epoch
    offset = pd.Period(pd.Timestamp(0), freq="D") - pd.Period(
        year=0, month=0, day=0, freq="D"
    )

    if timestamps:
        df["timestamp"] = pd.to_datetime(
            df["time_UTC"] - offset.n, unit="d"
        ) + pd.Timedelta(utc_offset, unit="h")
        # round to meaningful precision, in units of miliseconds
        df["timestamp"] = df["timestamp"].dt.round(str(int(1000 / sample_freq)) + "ms")
        first_timestamp = df["timestamp"][0]
    else:
        first_timestamp = pd.to_datetime(
            df["time_UTC"][0] - offset.n, unit="d"
        ) + pd.Timedelta(utc_offset, unit="h")
        first_timestamp = first_timestamp.round(str(int(1000 / sample_freq)) + "ms")

    # check date consistency
    file_date = pd.to_datetime(filepath.name[:16], format="%m_%d_%Y_%H_%M")
    if file_date != (first_timestamp - pd.Timedelta(utc_offset, unit="h")):
        raise ValueError(
            f"Timestamp mismatch. Filename {filepath.name} implies {file_date}, but first timestamp is {first_timestamp}"
        )

    if drop_raw_time:
        return df.drop(columns=["time_UTC"])
    else:
        return df


def sonic_summary(
    df: pd.DataFrame, height: int, eval_periods: Tuple[float, ...] = (60, 30, 10, 2)
) -> Dict[str, Union[float, int]]:
    """downsample high frequency sonic anemometer data by aggregating to industry standard metrics

    Args:
        df (pd.DataFrame): dataframe of sonic data, as loaded from matlab_to_pandas()
        height (int): height of anemometer, in meters
        eval_periods (Tuple[float, ...], optional): periods at which to evaluate the cumulative integral of power spectral density, in seconds. Defaults to (60, 30, 10, 2).

    Returns:
        Dict[str, Union[float, int]]: aggregate metrics
    """
    out: Dict[str, Union[float, int]] = {}
    out["height"] = height
    horizontal = f"Sonic_CupEqHorizSpeed_{height}m"
    vertical = f"Sonic_z_clean_{height}m"
    direction = f"Sonic_direction_{height}m"
    eval_freqs = tuple([1.0 / x for x in eval_periods])
    square_labels = [f"cum_square_sd_{period}s" for period in eval_periods]
    cube_labels = [f"cum_cube_sd_{period}s" for period in eval_periods]
    vert_square_labels = [f"vert_cum_square_sd_{period}s" for period in eval_periods]

    out["nan_count"] = df[horizontal].isna().sum()
    out["mean"] = df[horizontal].mean()
    out["diff_mean_sq"] = df[horizontal].var()
    out["diff_mean_cube"] = df[horizontal].pow(3).mean() - out["mean"] ** 3
    # square
    cum_sd = aggregate.integrated_spectral_densities(
        df[horizontal].values,
        total=(out["diff_mean_sq"]),
        eval_freqs=eval_freqs,
        square=True,
    )
    out.update(dict(zip(square_labels, cum_sd)))
    # cube
    cum_sd = aggregate.integrated_spectral_densities(
        df[horizontal].values,
        total=(out["diff_mean_cube"]),
        eval_freqs=eval_freqs,
        square=False,
    )
    out.update(dict(zip(cube_labels, cum_sd)))

    out["vert_nan_count"] = df[vertical].isna().sum()
    out["vert_mean"] = df[vertical].mean()
    out["vert_mean_square"] = df[vertical].pow(2).mean()
    out["vert_mean_cube"] = df[vertical].pow(3).mean()
    # vert square
    cum_sd = aggregate.integrated_spectral_densities(
        df[vertical].values, eval_freqs=eval_freqs, square=True,
    )
    out.update(dict(zip(vert_square_labels, cum_sd)))

    out["dir_mean"] = aggregate.direction_mean(df[direction])  # type: ignore
    out["waked_frac"] = (
        df[[direction]].query(f"80 < {direction} < 210").count()[0] / 12000
    )
    out["min"], out["max"] = aggregate.min_max(df[horizontal])
    out["3s_gust"] = aggregate.three_second_gust(df[horizontal], SONIC_SAMPLE_FREQ)  # type: ignore

    return out


def cup_summary(
    df: pd.DataFrame, inst: str, eval_periods: Tuple[float, ...] = (60, 30, 10, 2)
) -> Dict[str, Union[float, int]]:
    """downsample high frequency cup anemometer data by aggregating to industry standard metrics

    Args:
        df (pd.DataFrame): dataframe of meteorological data, as loaded from matlab_to_pandas()
        inst (str): cup anemometer name
        eval_periods (Tuple[float, ...], optional): periods at which to evaluate the cumulative integral of power spectral density, in seconds. Defaults to (60, 30, 10, 2).

    Returns:
        Dict[str, Union[float, int]]: aggregate metrics
    """
    out: Dict[str, Union[float, int]] = {}
    out["height"] = int(inst.split("_")[-1][:-1])  # names end with _123m
    sample_ratio = int(
        SONIC_SAMPLE_FREQ / CUP_SAMPLE_FREQ
    )  # cups are oversampled at 20Hz, actual data is 1Hz
    resampled = df[inst][::sample_ratio]
    eval_freqs = tuple([1.0 / x for x in eval_periods])
    square_labels = [f"cum_square_sd_{period}s" for period in eval_periods]
    cube_labels = [f"cum_cube_sd_{period}s" for period in eval_periods]

    out["nan_count"] = resampled.isna().sum()
    out["mean"] = resampled.mean()
    out["mean_square"] = resampled.pow(2).mean()
    out["mean_cube"] = resampled.pow(3).mean()
    # square
    cum_sd = aggregate.integrated_spectral_densities(
        resampled.values,
        total=(out["mean_square"] - out["mean"] ** 2),
        eval_freqs=eval_freqs,
        square=True,
        sample_freq=1,
    )
    out.update(dict(zip(square_labels, cum_sd)))
    # cube
    cum_sd = aggregate.integrated_spectral_densities(
        resampled.values,
        total=(out["mean_cube"] - out["mean"] ** 3),
        eval_freqs=eval_freqs,
        square=False,
        sample_freq=1,
    )
    out.update(dict(zip(cube_labels, cum_sd)))
    out["min"], out["max"] = aggregate.min_max(resampled)
    out["3s_gust"] = aggregate.three_second_gust(resampled, CUP_SAMPLE_FREQ)  # type: ignore
    return out


def misc_summary(df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """downsample high frequency meteorological data by aggregating to industry standard metrics

    Args:
        df (pd.DataFrame): dataframe of sonic data, as loaded from matlab_to_pandas()

    Returns:
        Dict[str, Union[float, int]]: aggregate metrics
    """
    out = {}
    out["mean_temp_38"] = df["Air_Temp_38m"].mean()
    out["mean_temp_122"] = (
        out["mean_temp_38"] + df["DeltaT_122_87m"].mean() + df["DeltaT_87_38m"].mean()
    )
    out["mean_precip"] = df["PRECIP_INTEN"].mean()
    return out
