from typing import List
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_tabular(
    x: np.ndarray,
    y: List[str],
    x_label: str = "Importance score",
    y_label: str = "Features",
    num_features: int = None,
    show_plot: bool = True,
    output_filename: Optional[str] = None,
) -> plt.Figure:
    """Plot feature importance with segments highlighted.

    Args:
        x (np.ndarray): Array of feature importance scores
        y (List[str]): List of feature names
        x_label (str): Label for the x-axis
        y_label (str): Label or list of labels for the y-axis
        num_features (int): Number of top features to display
        show_plot (bool, optional): Shows plot if true (for testing or writing
            plots to disk instead).
        output_filename (str, optional): Name of the file to save
            the plot to (optional).

    Returns:
        plt.Figure
    """
    if not num_features:
        num_features = len(x)
    fig = plt.figure()
    abs_values = [abs(i) for i in x]
    top_values = [x for _, x in sorted(zip(abs_values, x), reverse=True)][:num_features]
    top_features = [x for _, x in sorted(zip(abs_values, y), reverse=True)][
        :num_features
    ]

    colors = ["r" if x >= 0 else "b" for x in top_values]
    plt.barh(top_features, top_values, color=colors)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if show_plot:
        plt.show()
    if output_filename:
        plt.savefig(output_filename)

    return fig
