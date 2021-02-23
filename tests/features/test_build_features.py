import pandas as pd
import numpy as np

from src.features.build_features import angular_difference

def test_angular_difference():
    '''
    s1 =        pd.Series([  0,   0,     0,   0,      0, 50,  50, 50], dtype=float)
    s2 =        pd.Series([350,  10, 180.1, 180,  179.9, 40,  60, 50])
    expected =  pd.Series([ 10, -10, 179.9, 180, -179.9, 10, -10,  0])
    '''
    s1 = pd.Series([0, 0, 0, 0, 0, 50, 50, 50], dtype=float)
    s2 = pd.Series([350, 10, 180.1, 180, 179.9, 40, 60, 50])
    expected = pd.Series([10, -10, 179.9, 180, -179.9, 10, -10, 0])
    actual = angular_difference(s1, s2)
    message = f"Expected: {expected.to_list()}\nActual:   {actual.to_list()}"
    assert np.allclose(expected.to_numpy(), actual.to_numpy()), message
