from dianna.visualization import plot_timeseries
import numpy as np


def test_plot_timeseries():
    segments = [
        {'index': 0, 'start': 0, 'stop': 2, 'weight':-0.6},
        {'index': 1, 'start': 2, 'stop': 4, 'weight':-0.3},
        {'index': 2, 'start': 4, 'stop': 6, 'weight': 0.0},
        {'index': 3, 'start': 6, 'stop': 8, 'weight': 0.4},
        {'index': 4, 'start': 8, 'stop': 10, 'weight': 0.7},
    ]

    x = np.linspace(0, 10, 20)
    y = np.sin(x)

    plot_timeseries(x=x, y=y, segments=segments, show_plot=False)

