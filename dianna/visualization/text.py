from IPython.display import HTML
from IPython.display import display


def highlight_text(explanation,
                   original_text,
                   show_plot=True,
                   output_html_filename=None,
                   max_opacity=.8):
    """
    Highlights a given text based on values in a given explanation object.

    Args:
        explanation: list of tuples of (word, index of word in original data, importance)
        original_text: original text
        show_plot: Shows plot if true (for testing or writing plots to disk instead)
        output_html_filename: Name of the file to save the plot to (optional).
        max_opacity: Maximum opacity (0-1)

    Returns:
        None
    """
    output = _create_html(original_text, explanation, max_opacity)

    if output_html_filename:
        with open(output_html_filename, 'w', encoding='utf-8') as output_html_file:
            print(output, file=output_html_file)

    if show_plot:
        display(HTML(output))


def _create_html(original_text, explanation, max_opacity):
    max_importance = max([abs(item[2]) for item in explanation])
    body = original_text
    words_in_reverse_order = sorted(explanation, key=lambda item: item[1], reverse=True)
    for word, word_start, importance in words_in_reverse_order:
        word_end = word_start + len(word)
        highlighted_word = _highlight_word(word, importance, max_importance, max_opacity)
        body = body[:word_start] + highlighted_word + body[word_end:]
    return '<html><body>' + body + '</body></html>'


def _highlight_word(word, importance, max_importance, max_opacity):
    opacity = max_opacity * abs(importance) / max_importance
    if importance > 0:
        color = f'rgba(255, 0, 0, {opacity:.2f})'
    else:
        color = f'rgba(0, 0, 255, {opacity:2f})'
    highlighted_word = f'<span style="background:{color}">{word}</span>'
    return highlighted_word
