import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def highlight_text(explanation,
                   input_tokens=None,
                   show_plot=True,
                   output_filename=None,
                   colormap="RdBu",
                   alpha=1.0,
                   heatmap_range=(-1, 1)):
    """Highlights a given text based on values in a given explanation object.

    Args:
        explanation: list of tuples of (word, index of word in original data, importance)
        input_tokens: list of all tokens (including those without importance)
        show_plot: Shows plot if true (for testing or writing plots to disk instead)
        output_filename: Name of the file to save the plot to (optional).
        colormap: color map for the heatmap plot (see mpl.Axes.imshow documentation for options).
        heatmap_range: a tuple (vmin, vmax) to set the range of the heatmap.
    Returns:
        None
    """
    tokens, _, importances = zip(*explanation)

    if input_tokens:
        # Make a list of tuples (token, i, importance) for each token in the
        # input_tokens. if a token isnot in the explanantion, the importance is
        # None
        explanation = [
            (
                token, i, importances[tokens.index(token)] if token in tokens else None
            )
            for i, token in enumerate(input_tokens)
        ]

    vmin, vmax = heatmap_range

    x, y = (0, 0) # the initial position of the text
    space_token = ' '
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.axis('off')

    for token, _, importance in explanation:

        color = _get_text_color(importance, vmin, vmax, colormap, alpha)
        text = ax.text(x, y, token, fontsize=12, backgroundcolor=color)

        # Get the bounding box of the text in display space and convert it to data space
        bbox = text.get_window_extent()
        bbox_data = mtransforms.Bbox(ax.transData.inverted().transform(bbox))
        x = bbox_data.x1

        # Add a space after each token
        text = ax.text(x, y, space_token, fontsize=12)
        bbox = text.get_window_extent()
        bbox_data = mtransforms.Bbox(ax.transData.inverted().transform(bbox))

        # The next x is the right side of the bbox plus the fixed space
        x = bbox_data.x1

        # Wrap the text if token is a dot
        if token == '.':
            x = 0
            y -= 0.5  # space between lines in inches

    # adjust the height of the figure
    y_hight = 1
    if abs(y) > 1:
        y_hight = y
    ax.set_ylim(y_hight, 0)
    fig.set_figheight(abs(y_hight))

    # add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(colormap),
        norm=plt.Normalize(vmin, vmax)
        )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='horizontal', aspect=20, use_gridspec=True)
    # TODO add alpha to the colorbar

    if not show_plot:
        plt.close()

    if output_filename:
        plt.savefig(output_filename)

    return fig, ax


def _get_text_color(importance, vmin, vmax, colormap, alpha):
    if importance is None:
        return "none"

    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin, vmax)
    r, g, b, _ = cmap(norm(importance))
    return (r, g, b, alpha)