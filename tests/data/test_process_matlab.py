import pandas as pd
import numpy as np
import pytest

import src.data.process_matlab as pm


def test_wind_dir_from_vec():
    # clockwise around the unit square, steps of 45 degrees.
    # 0 degrees defined as North, so (x=0, y=1)
    points = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
    expected = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    actual = [pm.wind_dir_from_vec(*point) for point in points]
    message = (
        f"Tested 8 points around unit square. Expected: {expected}\nActual: {actual}"
    )
    assert actual == pytest.approx(expected), message


def test_direction_mean():
    test_data = {
        "normal": {
            "pairs": ((0, 90), (135, 225), (225, 315)),
            "expected": np.array([45.0, 180.0, 270.0]),
        },
        "wrap_0": {"pairs": ((315, 45),), "expected": np.array([0.0])},
        "cancellation": {"pairs": ((0, 180),), "expected": np.array([np.nan])},
    }

    for k, v in test_data.items():
        actual = [pm.direction_mean(pd.Series(pair)) for pair in v["pairs"]]
        expected = v["expected"]
        message = (
            f"{k} test pairs: {v['pairs']} \nExpected: {expected}\nActual: {actual}"
        )
        assert actual == pytest.approx(expected, nan_ok=True), message
