from typing import Dict
from _models_text import tokenizer


# Derived from: https://github.com/arnaudmiribel/streamlit-extras
# /blob/main/src/streamlit_extras/word_importances/__init__.py
def format_word_importances(text, importance_map: Dict[str, float]) -> str:
    """Adds a background color to each word based on its importance (float from -1 to 1).

    Args:
        text (str): input text
        importance_map (dict): dictionary with importances per word

    Returns:
        html: HTML string with formatted word
    """
    tokens = tokenizer.tokenize(text)

    max_importance = max(abs(val) for val in importance_map.values())

    tags = ['<td>']
    for token in tokens:
        importance = importance_map.get(token)

        if importance is None:
            bg_style = ''
        else:
            # normalize to max importance
            importance = importance / max_importance
            color = _get_color(importance)
            bg_style = f'background-color: {color};'

        unwrapped_tag = (
            f'<mark style="{bg_style}opacity:1.0;'
            f'        line-height:1.75"><font color="black"> {token}            '
            '       </font></mark>')
        tags.append(unwrapped_tag)

    tags.append('</td>')
    html = ''.join(tags)

    return html


def _get_color(importance: float) -> str:
    # clip values to prevent CSS errors (Values should be from [-1,1])
    importance = max(-1, min(1, importance))
    if importance > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * importance)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * importance)
    return f'hsl({hue}, {sat}%, {lig}%)'
