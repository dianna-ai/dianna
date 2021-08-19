import matplotlib.pyplot as plt


def determine_vmax(max_data_value):
    vmax = 1
    if max_data_value > 255:
        vmax = None
    elif max_data_value > 1:
        vmax = 255
    return vmax


def plot_image(heatmap, original_data=None, heatmap_cmap=None, data_cmap=None):
    """
    Example image figure
    """
    # default cmap depends on shape: grayscale or colour

    fig, ax = plt.subplots()
    alpha = 1
    if original_data is not None:
        if len(original_data.shape) == 2 and data_cmap is None:
            # 2D array, grayscale
            data_cmap = 'gray'

        ax.imshow(original_data, cmap=data_cmap, vmin=0, vmax=determine_vmax(original_data.max()))
        alpha = .5

    ax.imshow(heatmap, cmap=heatmap_cmap, alpha=alpha)
    plt.show()
