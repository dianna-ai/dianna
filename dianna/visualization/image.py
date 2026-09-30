from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from dianna.utils.misc import renamed_kwargs


def _determine_vmax(max_data_value):
    vmax = 1
    if max_data_value > 255:
        vmax = None
    elif max_data_value > 1:
        vmax = 255
    return vmax


@renamed_kwargs(heatmap='relevances', heatmap_cmap='relevances_cmap', heatmap_range='relevances_range')
def plot_image(relevances: np.ndarray,
               original_data: Optional[np.ndarray] = None,
               relevances_cmap='bwr',
               relevances_range=(None, None),  # (vmin, vmax)
               data_cmap=None,
               show_plot: bool = True,
               output_filename=None,
               ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots the relevance scores of an image explanation as a heatmap.

    Optionally, the heatmap can be plotted on top of the original data. In that case both images are plotted
    transparantly with alpha = 0.5.

    Args:
        relevances: the relevance scores (e.g. the output of an explainer) to be plotted.
        original_data: the data to plot together with the relevances, both with
                       alpha = 0.5 (optional).
        relevances_cmap: color map for the heatmap plot (see mpl.Axes.imshow
                         documentation for options).
        relevances_range: a tuple (vmin, vmax) to set the range of the heatmap.
                          By default, the colormap covers the complete value range of
                          the supplied relevances.
        data_cmap: color map for the (optional) data image (see mpl.Axes.imshow
                   documentation for options). By default, if the image is two
                   dimensional, the color map is set to 'gray'.
        show_plot: Shows plot if true (for testing or writing plots to disk
                   instead).
        output_filename: Name of the file to save the plot to (optional).
        ax: matplotlib.Axes object to plot on (optional).

    Returns:
        None
    """
    # default cmap depends on shape: grayscale or colour

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    alpha = 1
    if original_data is not None:
        if len(original_data.shape) == 2 and data_cmap is None:
            # 2D array, grayscale
            data_cmap = 'gray'

        ax.imshow(original_data,
                  cmap=data_cmap,
                  vmin=0,
                  vmax=_determine_vmax(original_data.max()))
        alpha = .5

    vmin, vmax = relevances_range
    cax = ax.imshow(relevances,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=relevances_cmap,
                    alpha=alpha)
    fig.colorbar(cax)
    ax.tick_params(bottom=False,
                   left=False,
                   right=False,
                   top=False,
                   labelleft=False,
                   labelbottom=False,
                   labelright=False,
                   labeltop=False)

    if not show_plot:
        plt.close()

    if output_filename:
        plt.savefig(output_filename)

    return fig, ax
