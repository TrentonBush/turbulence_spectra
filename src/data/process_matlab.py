import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.fft import next_fast_len, rfft
from scipy.integrate import cumtrapz
from typing import Tuple


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
    square=False,
    sample_freq: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # no overlap because
    #   1) I need a low number of segments to preserve resolution, and
    #   2) don't want to underweight information at the ends of the signal
    N = len(signal)
    nperseg = segment_size_seconds * sample_freq

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
    total: float = None,
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


def cup_summary(df: pd.DataFrame, inst: str):
    raise NotImplementedError  # need spectral density functions
    out = {}
    resampled = df[inst][::20]  # cups are oversampled at 20Hz, actual data is 1Hz

    out["nan_count"] = resampled.isna().sum()
    out["mean"] = resampled.mean()
    out["mean_square"] = resampled.pow(2).mean()
    out["mean_cube"] = resampled.pow(3).mean()

    return out


def misc_summary(df: pd.DataFrame):
    out = {}
    out["mean_delta_temp"] = df["DeltaT_122_87m"].mean() + df["DeltaT_87_38m"].mean()
    out["mean_temp"] = df["Air_Temp_38m"].mean()
    out["mean_precip"] = df["PRECIP_INTEN"].mean()
    return out
