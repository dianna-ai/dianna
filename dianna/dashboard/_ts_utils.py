import numpy as np


def open_timeseries(file):
    """Open a time series from a file and returns it as a numpy array."""
    return np.arange(10), np.arange(10)**2
