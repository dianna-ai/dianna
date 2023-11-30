import numpy as np


def open_timeseries(file):
    """Open a time series from a file and returns it as a numpy array."""
    return np.load(file)


def _convert_to_segments(explanation):
    """Convert explanation to segments."""
    import numpy as np

    def normalize(data):
        """Squash all values into [-1,1] range."""
        zero_to_one = (data - np.min(data)) / (np.max(data) - np.min(data))
        return 2 * zero_to_one - 1

    heatmap_channel = normalize(explanation[0])
    segments = []
    for i, val in enumerate(heatmap_channel):
        segments.append({
            'index': i,
            'start': i - 0.5,
            'stop': i + 0.5,
            'weight': val
        })

    return segments
