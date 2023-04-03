from _models_text import tokenizer
from dianna.visualization.text import _create_html


def format_word_importances(text, relevances) -> str:
    tokens = tokenizer.tokenize(text)
    return _create_html(tokens, relevances)
