"""Unit tests for visualization modules."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
from dianna.visualization import plot_tabular
from dianna.visualization import plot_timeseries
from dianna.visualization.text import highlight_text


def test_plot_tabular(tmpdir):
    """Test plot tabular data."""
    x = np.linspace(-5, 5, 3)
    y = [f"Feature {i}" for i in range(len(x))]
    output_path = Path(tmpdir) / "temp_visualization_test_tabular.png"

    plot_tabular(x=x, y=y, show_plot=False, output_filename=output_path)

    assert output_path.exists()

def test_plot_tabular_with_ndarray():
    x = np.random.rand(5, 3)
    y = [f"Feature {i}" for i in range(x.shape[1])]
    # check ValueError
    with pytest.raises(ValueError):
        plot_tabular(x=x, y=y, show_plot=False)


def test_plot_timeseries_univariate(tmpdir, random):
    """Test plot univariate time series."""
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    segments = get_test_segments(data=np.expand_dims(y, 0))

    output_path = Path(tmpdir) / "temp_visualization_test_univariate.png"

    plot_timeseries(x=x,
                    y=y,
                    segments=segments,
                    show_plot=False,
                    output_filename=output_path)

    assert output_path.exists()


def test_plot_timeseries_multivariate(tmpdir, random):
    """Test plot multivariate time series."""
    x = np.linspace(start=0, stop=10, num=20)
    ys = np.stack((np.sin(x), np.cos(x), np.tan(0.4 * x)))
    segments = get_test_segments(data=ys)
    output_path = Path(tmpdir) / "temp_visualization_test_multivariate.png"

    plot_timeseries(x=x,
                    y=ys.T,
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
            segment = {
                "index": i_segment + i_channel * n_segments,
                "start": i_segment,
                "stop": i_segment + 1,
                "weight": data[i_channel, factor * i_segment],
            }
            if n_channels > 1:
                segment["channel"] = i_channel
            segments.append(segment)

    return segments


@pytest.fixture
def random():
    """Set the random seed."""
    np.random.seed(0)


class TestHighlightText:
    """Test highlight text function."""
    def test_highlight_text_without_input_token(self, tmp_path):
        """Test highlight text without input tokens."""
        explanation = [("Hello", 0, 0.5), ("world", 1, -0.5)]
        output_path = Path(tmp_path) / "hello_world.png"

        fig, ax = highlight_text(explanation=explanation,
                                output_filename=output_path)

        assert output_path.exists()
        assert fig is not None
        assert ax is not None
        assert ax.texts[0].get_text() == "Hello"
        assert ax.texts[1].get_text() == " "
        assert ax.texts[2].get_text() == "world"

        # Get the colorbar and check range
        cbar = fig.axes[-1]
        assert cbar.get_xlim() == (-1, 1)

    def test_highlight_text_with_input_token(self):
        """Test highlight text with input tokens."""
        explanation = [("Hello", 0, 0.5), ("world", 1, -0.5)]
        input_tokens = ["Hello", "world", "!", "This", "is", "a", "test"]

        fig, ax = highlight_text(explanation=explanation,
                                input_tokens=input_tokens)

        assert fig is not None
        assert ax is not None
        assert ax.texts[0].get_text() == "Hello"
        assert ax.texts[1].get_text() == " "
        assert ax.texts[2].get_text() == "world"
        assert ax.texts[3].get_text() == " "
        assert ax.texts[4].get_text() == "!"

    def test_highlight_text_with_range(self):
        """Test highlight text with heatmap range."""
        explanation = [("Hello", 0, 0.5), ("world", 1, -0.5)]

        fig, ax = highlight_text(explanation=explanation,
                                heatmap_range=(0, 1))

        assert fig is not None
        assert ax is not None
        # Get the colorbar and check range
        cbar = fig.axes[-1]
        assert cbar.get_xlim() == (0, 1)

    def test_highlight_text_show_plot_false(self):
        """Test highlight text with show plot false."""
        explanation = [("Hello", 0, 0.5), ("world", 1, -0.5)]

        fig, ax = highlight_text(explanation=explanation, show_plot=False)

        assert fig is not None
        assert ax is not None

        # check that the plot is closed
        assert not plt.fignum_exists(fig.number)

    def test_highlight_text_ends_with_dots(self):
        """Test highlight multiline text with dots in text."""
        explanation = [
            ("Hello", 0, 0.5), ("world", 1, -0.5), (".", 2, 0.0),
            ("This", 3, 0.5), ("is", 4, -0.5), ("a", 5, 0.0),
            ("test", 6, 0.5), (".", 7, -0.5),
            ("Another", 8, 0.0), ("test", 9, 0.5), (".", 10, -0.5),
            ]

        fig, ax = highlight_text(explanation=explanation)

        assert fig is not None
        assert ax is not None
