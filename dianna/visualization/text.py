def highlight_text(heatmap, original_data=None, heatmap_cmap=None, show_plot=True, output_img_filename=None, output_html_filename=None):  # pylint: disable=too-many-arguments
    """
    Highlights text
    Args:
        output_img_filename:
        output_html_filename:
        heatmap:
        original_data:
        heatmap_cmap:
        data_cmap:
        show_plot: Shows plot if true (for testing or writing plots to disk instead)
        output_filename: Name of the file to save the plot to (optional).

    Returns:
        None
    """
    output = original_data
    if output_html_filename:
        with open(output_html_filename, 'w') as output_html_file:
            print(output, file=output_html_file)
