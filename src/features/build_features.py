import pandas as pd
import numpy as np

def angular_difference(series1, series2):
    """Take two angular vectors (in degrees), each in [0, 360), and express their difference in (-180, 180]
    output = series1 - series2

    Parameters
    ----------
    series1 : pd.Series
        vector of angular directions, in degrees
    series2 : pd.Series
        vector of angular directions, in degrees

    Returns
    -------
    pd.Series
        differences, in (-180, 180]
    """
    return ((series1 - series2) - 180) % -360 + 180

def power_law_shear(upper_speed: pd.Series, lower_speed: pd.Series, upper_height: float, lower_height: float) -> pd.Series:
    """Take two wind speed vectors and calculate the power law exponent. Length units must be compatible - either all meters and m/s or feet and ft/s
    power law model: upper_speed / upper_height = (lower_speed / lower_height)^exponent

    Parameters
    ----------
    upper_speed : pd.Series
        vector of wind speeds at upper height
    lower_speed : pd.Series
        vector of wind speeds at lower height
    upper_height : float
        height of upper anemometer
    lower_height : float
        height of lower anemometer

    Returns
    -------
    pd.Series
        power law exponents, alpha
    """
    const = np.log(upper_height / lower_height)
    return np.log(upper_speed / lower_speed) / const
