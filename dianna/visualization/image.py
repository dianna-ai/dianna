import matplotlib.pyplot as plt


def plot_image(heatmap, original_data=None, cmap=None):
    """
    Example image figure
    """
    # default cmap depends on shape: grayscale or colour
    if len(heatmap.shape) == 2 and cmap is None:
        # 2D array, grayscale
        cmap = 'gray'

    fig, ax = plt.subplots()
    alpha = 1
    if original_data is not None:
        ax.imshow(original_data, cmap=cmap, vmin=0, vmax=1)
        alpha = .5

    ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    plt.show()
