import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.fft import next_fast_len, rfft
from scipy.integrate import cumtrapz
from typing import Tuple, Optional, Sequence


EXPECTED_UTC_OFFSET = -7
STANDARD_SUBSET = (
    "Air_Temp_38m",
    "DeltaT_122_87m",
    "DeltaT_87_38m",
    "DeltaT_38_3m",
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


def matlab_to_pandas(
    filepath: Path, timestamps=False, col_subset: Optional[Sequence] = STANDARD_SUBSET
) -> pd.DataFrame:
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
    if utc_offset != EXPECTED_UTC_OFFSET:
        raise ValueError(f"Unexpected utc_offset: {utc_offset}.")

    df = pd.DataFrame()
    if col_subset is None:  # get all 150 cols
        for key in matlab.keys():
            if key not in exclusions:
                df[key] = matlab[key]["val"].item()
    else:
        for key in col_subset:
            df[key] = matlab[key]["val"].item()

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
        df["timestamp"] = df["timestamp"].round(str(int(1000 / sample_freq)) + "ms")
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

    return df


def cube_spectral_density(
    signal: np.ndarray, square=False, sample_freq: float = 20.0,
) -> np.ndarray:
    N = len(signal)
    if square:
        exponent = 2
    else:
        exponent = 3
    coeffs = rfft(signal, norm=None, n=next_fast_len(N)) / np.sqrt(N)
    coeffs = np.power(np.absolute(coeffs), exponent) / sample_freq
    if not square:
        coeffs /= np.sqrt(
            2
        )  # extra factor of 1/sqrt(2). Not sure derivation but confirmed vs manual test of E[x^3] - E[x]^3

    # one-sidedness correction due to rfft omitting negative frequencies. Explanation here: https://www.reddit.com/r/matlab/comments/4cqa10/fft_dc_component_scaling/
    if N % 2:  # odd, no nyquist coeff
        coeffs[1:] = coeffs[1:] * 2
    else:  # even, don't double Nyquist (last) coeff
        coeffs[1:-1] = coeffs[1:-1] * 2

    # remove meaningless DC term. DC component is just sum(signal), which then gets cubed to a meaningless value. Have to remove in downstream integration anyway.
    coeffs[0] = 0

    return coeffs


def cube_welch(
    signal: np.ndarray,
    segment_size_seconds: int = 150,
    square: bool = False,
    sample_freq: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # no overlap because
    #   1) I need a low number of segments to preserve resolution, and
    #   2) don't want to underweight information at the ends of the signal
    N = len(signal)
    nperseg = int(segment_size_seconds * sample_freq)

    n_chunks = N / nperseg
    if n_chunks % 1 != 0:
        raise ValueError(
            f"nperseg ({nperseg}) does not evenly divide N ({N}). Result is {n_chunks}"
        )
    n_chunks = int(n_chunks)

    indicies = np.arange(n_chunks + 1) * nperseg
    chunks = [
        cube_spectral_density(
            signal[indicies[i] : indicies[i + 1]],
            square=square,
            sample_freq=sample_freq,
        )
        for i in range(n_chunks)
    ]
    coeffs = np.vstack(chunks).mean(axis=0)

    freq_resolution = sample_freq / nperseg
    freqs = np.arange(coeffs.shape[0]) * freq_resolution

    return freqs, coeffs


def integrated_spectral_densities(
    anemometer: np.ndarray,
    total: Optional[float] = None,
    eval_freqs: Tuple[float, ...] = (1 / 60, 1 / 30, 1 / 10, 1 / 2),
    **kwargs,
) -> np.ndarray:
    if total is None:
        if kwargs.get("square"):  # square
            total = np.var(anemometer)
        else:  # cube
            total = np.mean(np.power(anemometer, 3)) - np.power(np.mean(anemometer), 3)

    freqs, coeffs = cube_welch(anemometer, **kwargs)
    coeffs = cumtrapz(coeffs, dx=(freqs[1] - freqs[0]))  # cumulative integral
    # add in missing power from unresolved low freqs
    coeffs = coeffs + (total - coeffs[-1])

    # evaluate at frequencies of interest
    return np.interp(eval_freqs, freqs[1:], coeffs)


def sonic_summary(
    df: pd.DataFrame, height: int, eval_periods: Tuple[float, ...] = (60, 30, 10, 2)
):
    out = {}
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
    cum_sd = integrated_spectral_densities(
        df[horizontal].values,
        total=(out["diff_mean_sq"]),
        eval_freqs=eval_freqs,
        square=True,
    )
    out.update(dict(zip(square_labels, cum_sd)))
    # cube
    cum_sd = integrated_spectral_densities(
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
    cum_sd = integrated_spectral_densities(
        df[vertical].values, eval_freqs=eval_freqs, square=True,
    )
    out.update(dict(zip(vert_square_labels, cum_sd)))

    out["dir_mean"] = df[direction].mean()
    out["waked_frac"] = (
        df[[direction]].query(f"80 < {direction} < 210").count()[0] / 12000
    )

    return out


def cup_summary(
    df: pd.DataFrame, inst: str, eval_periods: Tuple[float, ...] = (60, 30, 10, 2)
):
    out = {}
    out["height"] = int(inst.split("_")[-1][:-1])  # names end with _123m
    resampled = df[inst][::20]  # cups are oversampled at 20Hz, actual data is 1Hz
    eval_freqs = tuple([1.0 / x for x in eval_periods])
    square_labels = [f"cum_square_sd_{period}s" for period in eval_periods]
    cube_labels = [f"cum_cube_sd_{period}s" for period in eval_periods]

    out["nan_count"] = resampled.isna().sum()
    out["mean"] = resampled.mean()
    out["mean_square"] = resampled.pow(2).mean()
    out["mean_cube"] = resampled.pow(3).mean()
    # square
    cum_sd = integrated_spectral_densities(
        resampled.values,
        total=(out["mean_square"] - out["mean"] ** 2),
        eval_freqs=eval_freqs,
        square=True,
        sample_freq=1,
    )
    out.update(dict(zip(square_labels, cum_sd)))
    # cube
    cum_sd = integrated_spectral_densities(
        resampled.values,
        total=(out["mean_cube"] - out["mean"] ** 3),
        eval_freqs=eval_freqs,
        square=False,
        sample_freq=1,
    )
    out.update(dict(zip(cube_labels, cum_sd)))

    return out


def misc_summary(df: pd.DataFrame):
    out = {}
    out["mean_temp_38"] = df["Air_Temp_38m"].mean()
    out["mean_temp_122"] = (
        out["mean_temp_38"] + df["DeltaT_122_87m"].mean() + df["DeltaT_87_38m"].mean()
    )
    out["mean_precip"] = df["PRECIP_INTEN"].mean()
    return out
