from pathlib import Path
import numpy as np
import pytest
from dianna.visualization import plot_timeseries


def test_plot_timeseries_univariate(tmpdir, random):
    """Test plot univariate time series."""
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    segments = get_test_segments(data=np.expand_dims(y, 0))

    output_path = Path(tmpdir) / 'temp_visualization_test_univariate.png'

    plot_timeseries(x=x,
                    y=y,
                    segments=segments,
                    show_plot=False,
                    output_filename=output_path)

    assert output_path.exists()


def test_plot_timeseries_multivariate(tmpdir, random):
    """Test plot multivariate time series."""
    x = np.linspace(start=0, stop=10, num=20)
    y = np.stack((np.sin(x), np.cos(x), np.tan(0.4 * x)))
    segments = get_test_segments(data=y)
    output_path = Path(tmpdir) / 'temp_visualization_test_multivariate.png'

    plot_timeseries(x=x,
                    y=y,
                    segments=segments,
                    show_plot=False,
                    output_filename=output_path)

    assert output_path.exists()


def get_test_segments(data):
    """Creates some segments for testing the timeseries visualization."""
    n_channels = data.shape[0]
    n_steps = data.shape[1]
    factor = 2
    n_segments = n_steps // factor

    segments = []
    for i_segment in range(n_segments):
        for i_channel in range(n_channels):
            segments.append({
                'index': i_segment + i_channel * n_segments,
                'start': i_segment,
                'stop': i_segment + 1,
                'weight': data[i_channel, factor * i_segment],
                'channel': i_channel,
            })

    return segments


@pytest.fixture
def random():
    """Set the random seed."""
    np.random.seed(0)
