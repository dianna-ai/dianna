from IPython.display import HTML
from IPython.display import display


def highlight_text(explanation,
                   input_tokens,
                   show_plot=True,
                   output_html_filename=None,
                   max_opacity=.8):
    """Highlights a given text based on values in a given explanation object.

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
        with open(output_html_filename, 'w',
                  encoding='utf-8') as output_html_file:
            print(output, file=output_html_file)

    if show_plot:
        display(HTML(output))


def _create_html(tokens, explanation, opacity: float = 0.8):
    importance_map = {r[0]: r[2] for r in explanation}

    max_importance = max(abs(val) for val in importance_map.values())

    tags = []
    for token in tokens:
        importance = importance_map.get(token)

        if importance is None:
            color = f'hsl(0, 0%, 75%, {opacity})'
        else:
            # normalize to max importance
            importance = importance / max_importance
            color = _get_color(importance, opacity)

        tag = (f'<mark style="background-color: {color}; '
               f'line-height:1.75">{token}</mark>')
        tags.append(tag)

    html = ' '.join(tags)

    return html


def _get_color(importance: float, opacity: float) -> str:
    # clip values to prevent CSS errors (Values should be from [-1,1])
    importance = max(-1, min(1, importance))
    if importance > 0:
        hue = 0
        sat = 100
        lig = 100 - int(50 * importance)
    else:
        hue = 240
        sat = 100
        lig = 100 - int(-50 * importance)
    return f'hsl({hue}, {sat}%, {lig}%, {opacity})'
