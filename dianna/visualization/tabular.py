"""Visualization module for tabular data."""
from typing import List
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_tabular(
    x: np.ndarray,
    y: List[str],
    x_label: str = "Importance score",
    y_label: str = "Features",
    num_features: Optional[int] = None,
    show_plot: Optional[bool] = True,
    output_filename: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot feature importance with segments highlighted.

    Args:
        x (np.ndarray): 1D array of feature importance scores of one instance
        y (List[str]): List of feature names
        x_label (str): Label for the x-axis
        y_label (str): Label or list of labels for the y-axis
        num_features (Optional[int]): Number of most salient features to display
        show_plot (bool, optional): Shows plot if true (for testing or writing
            plots to disk instead).
        output_filename (str, optional): Name of the file to save
            the plot to (optional).
        ax (matplotlib.Axes, optional): externally created canvas to plot on.

    Returns:
        plt.Figure
    """
    # check type and shape of x should be 1D array
    if not isinstance(x, np.ndarray):
        raise TypeError("x should be a numpy array")
    if x.ndim != 1:
        raise ValueError("x should be a 1D array")

    if not num_features:
        num_features = len(x)
    abs_values = [abs(i) for i in x]
    top_values = [x for _, x in sorted(zip(abs_values, x), reverse=True)][:num_features]
    top_features = [x for _, x in sorted(zip(abs_values, y), reverse=True)][
        :num_features
    ]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    colors = ["r" if x >= 0 else "b" for x in top_values]
    ax.barh(top_features, top_values, color=colors)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if not show_plot:
        plt.close()

    if output_filename:
        plt.savefig(output_filename)

    return fig, ax
