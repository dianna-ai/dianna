import numpy as np


def open_timeseries(file):
    """Open a time series from a file and returns it as a numpy array."""
    return np.load(file)


def _convert_to_segments(explanation):
    """Convert explanation to segments."""
    segments = []
    for channel_number, explanation_channel in enumerate(explanation):
        for i, val in enumerate(explanation_channel):
            segments.append({
                'index': i,
                'start': i - 0.5,
                'stop': i + 0.5,
                'weight': val,
                'channel': channel_number
            })

    return segments


def _downsample_channels(data, factor):
    assert data.shape[
        0] % factor == 0, 'Downsampling factor must be a factor of the number of channels'
    return data.reshape(data.shape[0] // factor, factor,
                        *data.shape[1:]).mean(axis=1)
