import matplotlib.pyplot as plt


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

        ax.imshow(original_data, cmap=data_cmap, vmin=0, vmax=1)
        alpha = .5

    ax.imshow(heatmap, cmap=heatmap_cmap, alpha=alpha, vmin=0, vmax=1)
    plt.show()
