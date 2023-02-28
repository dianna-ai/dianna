import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from matplotlib import cm
import numpy as np


def plot_timeseries(
    x: np.ndarray,
    y: np.ndarray,
    segments: List[Dict[str, Any]],
    xlabel='x',
    ylabel='y',
    cmap: Optional[str] = None,
    show_plot: bool = False,
    output_filename: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
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
        xlabel (str, optional): Label for the x-axis
        ylabel (str, optional): Label for the y-axis
        cmap (str, optional): Matplotlib colormap
        show_plot (bool, optional): Shows plot if true (for testing or writing
            plots to disk instead).
        output_filename (str, optional): Name of the file to save
            the plot to (optional).
        ax (plt.Axes, optional): Matplotlib axes object
    """
    assert len(x) == len(y)

    if not ax:
        _, ax = plt.subplots()

    ax.plot(x, y, label='Timeseries')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cmap = plt.get_cmap(cmap)

    norm = plt.Normalize(-1, 1)

    for segment in segments:
        start = segment['start']
        stop = segment['stop']
        weight = segment['weight']
        index = segment['index']

        color = cmap(norm(weight))

        ax.axvspan(start, stop, color=color, alpha=0.5)
        ax.text(start, max(y), str(index))

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax,
                 label='weights')

    if show_plot:
        plt.show()
    if output_filename:
        plt.savefig(output_filename)

    return ax
