import matplotlib.pyplot as plt


def _determine_vmax(max_data_value):
    vmax = 1
    if max_data_value > 255:
        vmax = None
    elif max_data_value > 1:
        vmax = 255
    return vmax


def plot_image(heatmap, original_data=None, heatmap_cmap=None, data_cmap=None, show_plot=True, output_filename=None):  # pylint: disable=too-many-arguments
    """
    Plots a heatmap image.

    Optionally, the heatmap (typically a saliency map of an explainer) can be
    plotted on top of the original data. In that case both images are plotted
    transparantly with alpha = 0.5.

    Args:
        heatmap: the saliency map or other heatmap to be plotted.
        original_data: the data to plot together with the heatmap, both with
                       alpha = 0.5 (optional).
        heatmap_cmap: color map for the heatmap plot (see mpl.Axes.imshow
                      documentation for options).
        data_cmap: color map for the (optional) data image (see mpl.Axes.imshow
                   documentation for options). By default, if the image is two
                   dimensional, the color map is set to 'gray'.
        show_plot: Shows plot if true (for testing or writing plots to disk
                   instead).
        output_filename: Name of the file to save the plot to (optional).

    Returns:
        None
    """
    # default cmap depends on shape: grayscale or colour

    _, ax = plt.subplots()
    alpha = 1
    if original_data is not None:
        if len(original_data.shape) == 2 and data_cmap is None:
            # 2D array, grayscale
            data_cmap = 'gray'

        ax.imshow(original_data, cmap=data_cmap, vmin=0, vmax=_determine_vmax(original_data.max()))
        alpha = .5

    ax.imshow(heatmap, cmap=heatmap_cmap, alpha=alpha)
    if show_plot:
        plt.show()
    if output_filename:
        plt.savefig(output_filename)
