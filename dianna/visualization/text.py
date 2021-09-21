def highlight_text(explanation, original_data, show_plot=True, output_img_filename=None,    # pylint: disable=too-many-arguments
                   output_html_filename=None, max_opacity=.8):
    """
    Highlights text
    Args:
        output_img_filename:
        output_html_filename:
        explanation: list of tuples of (word, index of word in original data, importance)
        original_data: original text
        heatmap_cmap:
        data_cmap:
        show_plot: Shows plot if true (for testing or writing plots to disk instead)
        output_filename: Name of the file to save the plot to (optional).
        max_opacity: Maximum opacity (0-1)

    Returns:
        None
    """
    max_importance = max([abs(item[2]) for item in explanation])

    output = '<html><body>'
    current_char = 0
    for word, word_start, importance in sorted(explanation, key=lambda item: item[1]):
        output = output + original_data[current_char:word_start]

        opacity = max_opacity * abs(importance) / max_importance
        if importance > 0:
            color = f'rgba(255, 0, 0, {opacity:.2f})'
        else:
            color = f'rgba(0, 0, 255, {opacity:2f})'
        output = output + f'<span style="background:{color}">{word}</span>'
        current_char = word_start + len(word)

    if current_char < len(original_data):
        output += original_data[current_char:]

    output = output + '</body></html>'

    if output_html_filename:
        with open(output_html_filename, 'w', encoding='utf-8') as output_html_file:
            print(output, file=output_html_file)

    if output_img_filename:
        pass

    if show_plot:
        pass
