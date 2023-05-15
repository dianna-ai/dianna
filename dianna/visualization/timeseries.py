from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_timeseries(
    x: np.ndarray,
    y: np.ndarray,
    segments: List[Dict[str, Any]],
    x_label: str = 't',
    y_label: Union[str, Iterable[str]] = None,
    cmap: Optional[str] = None,
    show_plot: bool = False,
    output_filename: Optional[str] = None,
) -> plt.Axes:
    """Plot timeseries with segments highlighted.

    Args:
        x (np.ndarray): X-values
        y (np.ndarray): Y-values
        segments (List[Dict[str, Any]]): Segment data, must be a list of
            dicts with the following keys: 'index', 'start', 'end',
            'weight'. Here, `index` is the index of the segment of feature,
            `start` and `end` determine the location of the
            segment, and `weight` determines the color.
        x_label (str, optional): Label for the x-axis
        y_label (str, optional): Label for the y-axis
        cmap (str, optional): Matplotlib colormap
        show_plot (bool, optional): Shows plot if true (for testing or writing
            plots to disk instead).
        output_filename (str, optional): Name of the file to save
            the plot to (optional).

    Returns:
        plt.Figure
    """
    fig, axs, y_labels, ys = _process_plotting_parameters(y, y_label)

    for y_current, y_label_current, ax_current in zip(ys, y_labels, axs):
        current_ax = ax_current
        current_ax.plot(x, y_current, label=y_label_current)
        current_ax.set_xlabel(x_label)
        current_ax.set_ylabel(y_label_current)
        current_ax.label_outer()

    _draw_segments(axs, cmap, segments)

    if show_plot:
        plt.show()
    if output_filename:
        plt.savefig(output_filename)

    return fig


def _draw_segments(axs, cmap, segments):
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(-1, 1)
    for segment in segments:
        start = segment['start']
        stop = segment['stop']
        weight = segment['weight']
        segment['index']
        channel = segment['channel']

        color = cmap(norm(weight))

        axs[channel].axvspan(start, stop, color=color, alpha=0.5)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axs,
                 label='weights')


def _process_plotting_parameters(y, y_labels):
    if y.ndim == 1:
        print(y.shape)
        ys = np.expand_dims(y, 0)
        print(y.shape)
    elif y.ndim == 2:
        ys = y
    else:
        raise ValueError(
            f'Invalid rank {y.ndim}. Data y can only have either 1 or 2 dimensions.'
        )

    if not y_labels:
        y_labels = [f'channel {c}' for c in range(ys.shape[0])]
    if isinstance(y_labels, str):
        y_labels = [y_labels]

    n_channels = ys.shape[0]
    fig, ax = plt.subplots(nrows=n_channels, sharex=True)
    if n_channels == 1:
        axs = (ax, )
    else:
        axs = ax
    return fig, axs, y_labels, ys
