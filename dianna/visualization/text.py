from IPython.display import HTML
from IPython.display import display


def highlight_text(explanation,
                   input_tokens,
                   show_plot=True,
                   output_html_filename=None,
                   max_opacity=.8):
    """
    Highlights a given text based on values in a given explanation object.

    Args:
        explanation: list of tuples of (word, index of word in original data, importance)
        input_tokens: list of all tokens (including those without importance)
        show_plot: Shows plot if true (for testing or writing plots to disk instead)
        output_html_filename: Name of the file to save the plot to (optional).
        max_opacity: Maximum opacity (0-1)

    Returns:
        None
    """
    output = _create_html(input_tokens, explanation, max_opacity)

    if output_html_filename:
        with open(output_html_filename, 'w', encoding='utf-8') as output_html_file:
            print(output, file=output_html_file)

    if show_plot:
        display(HTML(output))


def _create_html(input_tokens, explanation, max_opacity):
    max_importance = max(abs(item[2]) for item in explanation)
    explained_indices = [index for _, index, _ in explanation]
    highlighted_words = []
    for index, word in enumerate(input_tokens):
        # if word has an explanation, highlight based on that, otherwise
        # make it grey
        try:
            explained_index = explained_indices.index(index)
            importance = explanation[explained_index][2]
            highlighted_words.append(
                _highlight_word(word, importance, max_importance, max_opacity)
                )
        except ValueError:
            highlighted_words.append(f'<span style="background:rgba(128, 128, 128, 0.3)">{word}</span>')

    return '<html><body>' + ' '.join(highlighted_words) + '</body></html>'


def _highlight_word(word, importance, max_importance, max_opacity):
    opacity = max_opacity * abs(importance) / max_importance
    if importance > 0:
        color = f'rgba(255, 0, 0, {opacity:.2f})'
    else:
        color = f'rgba(0, 0, 255, {opacity:2f})'
    highlighted_word = f'<span style="background:{color}">{word}</span>'
    return highlighted_word
