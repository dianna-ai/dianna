import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from matplotlib import cm
import numpy as np


def plot_timeseries(
    x: np.ndarray,
    y: np.ndarray,
    segments: List[Tuple[int, float]],
    xlabel='x',
    ylabel='y',
    cmap: Optional[str] = None,
    show_plot: bool = False,
    output_filename: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
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
        feature = segment['feature']

        color = cmap(norm(weight))

        ax.axvspan(start, stop, color=color, alpha=0.5)
        ax.text(start, max(y), str(feature))

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax,
                 label='weights')

    if show_plot:
        plt.show()
    if output_filename:
        plt.savefig(output_filename)

    return ax
