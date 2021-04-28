"""aggregate high frequency data to low frequency metrics"""
import pandas as pd
import numpy as np
from scipy.fft import next_fast_len, rfft
from scipy.integrate import cumtrapz
from typing import Tuple, Optional


def cube_spectral_density(
    signal: np.ndarray, square=False, sample_freq: float = 20.0,
) -> np.ndarray:
    """calculate custom power spectral density with cubed or squared coefficients

    Args:
        signal (np.ndarray): input samples. No gaps or NaNs.
        square (bool, optional): whether to square or cube the coefficients. Defaults to False.
        sample_freq (float, optional): sample frequency of input signal, in Hz. Defaults to 20.0.

    Returns:
        np.ndarray: cubed (or squared) and normalized DFT coefficients
    """
    N = len(signal)
    if square:
        exponent = 2
    else:
        exponent = 3
    coeffs = rfft(signal, norm="backward", n=next_fast_len(N)) / np.sqrt(N)
    coeffs = np.power(np.absolute(coeffs), exponent) / sample_freq
    if not square:
        # rfft default normalization doesn't anticipate cubing, so has an extra factor of sqrt(2)
        # Not sure derivation but confirmed via manual test of E[x^3] - E[x]^3
        coeffs /= np.sqrt(2)

    # one-sidedness correction due to rfft omitting negative frequencies.
    # Explanation here: https://www.reddit.com/r/matlab/comments/4cqa10/fft_dc_component_scaling/
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
    """apply the Welch method to estimate PSD with either cubed or squared coeffs

    Args:
        signal (np.ndarray): input signal
        segment_size_seconds (int, optional): size of sub-segments used in Welch method, in seconds. Defaults to 150.
        square (bool, optional): whether to square or cube the coeffs. Defaults to False.
        sample_freq (float, optional): sample frequency of signal, in Hz. Defaults to 20.0.

    Raises:
        ValueError: if signal does not divide evenly into segments of size segment_size_seconds * sammple_freq

    Returns:
        Tuple[np.ndarray, np.ndarray]: arrays of frequencies and coefficients
    """
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
    """Calculate custom PSD coefficients and integrate them up to each cutoff frequency in eval_freqs

    Args:
        anemometer (np.ndarray): input signal of wind speeds
        total (Optional[float], optional): tune the integrated PSD to this total energy by adding an offset. Defaults to None.
        eval_freqs (Tuple[float, ...], optional): cutoff frequencies to integrate up to. Defaults to (1 / 60, 1 / 30, 1 / 10, 1 / 2).

    Returns:
        np.ndarray: integrated energy at each eval_freq
    """
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


def min_max(column: pd.Series) -> tuple:
    """get the min and max values of a series"""
    return tuple(column.quantile([0, 1]).squeeze())


def three_second_gust(column: pd.Series, sample_freq: float) -> float:
    """calculate three second gust, a commonly available metric used to assess structural strength requirements"""
    window_size = 3 * sample_freq
    return column.rolling(window_size).mean().max()


def wind_dir_from_vec(x: float, y: float) -> np.float64:
    """Calculate wind direction angle from vector components

    NOTE: if calculating direction from wind velocity components,
    the components must be multiplied by -1 before passing into this function.
    This is due to wind direction sign convention (defined as
    where wind is ~~coming from~~, not where it is going)

    Wind direction angle is defined as clockwise degrees from North,
    which is directionally reversed and 90 degrees rotated from 'normal' angles on an xy plane.

    Parameters
    ----------
    x : float
        x vector component
    y : float
        y vector component

    Returns
    -------
    np.float64
        angle in interval [0, 360) or nan, if given input (0,0)
    """
    # fix edge case of (0,0)
    x_is_zero = np.abs(x) < 1e-12
    y_is_zero = np.abs(y) < 1e-12
    if x_is_zero & y_is_zero:
        return np.nan

    angle = np.arctan2(y, x)  # function signature is (y, x)
    angle *= 180 / np.pi * -1  # convert counterclockwise radians to clockwise degrees
    angle += 90  # rotate reference axis from (x=1, y=0) to North (x=0, y=1)
    angle = np.round(angle, 12) % 360  # convert interval from (-180, 180] to [0, 360)
    # I have to round before mod 360 because negative epsilon % 360 -> 360, which violates my desired bounds.
    # It's confusing, but trust the tests! See tests/data/test_aggregate.py

    return angle


def mean_vec_from_dirs(direction: pd.Series) -> Tuple[float, float]:
    """Component-wise mean of wind direction"""
    dir_radians = direction * (np.pi / 180)
    # y corresponds to cosine because wind_direction is defined as clockwise degrees from North (x=0, y=1)
    # It's confusing, but trust the tests! See tests/data/test_aggregate.py
    y = np.cos(dir_radians)
    x = np.sin(dir_radians)
    return (x.mean(), y.mean())


def direction_mean(direction: pd.Series,) -> float:
    """Calculate mean of wind direction angle by projecting component-wise mean

    Parameters
    ----------
    direction : pd.Series
        series of wind directions in [0, 360)

    Returns
    -------
    float
        mean direction, in [0, 360)
    """
    mean_vector = mean_vec_from_dirs(direction)
    return wind_dir_from_vec(*mean_vector)


def matlab_filename_from_timestamp(timestamp: pd.Timestamp, utc_offset: int = -7) -> str:
    """create an NREL matlab filename from the given timestamp and UTC offset"""
    return (timestamp - pd.Timedelta(utc_offset, unit='hours')).strftime("%m_%d_%Y_%H_%M_%S_000.mat")
