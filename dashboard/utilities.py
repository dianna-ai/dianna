import plotly.graph_objects as go
from dash import html
import numpy as np
from PIL import Image, ImageStat
from dianna import utils
from torchtext.data import get_tokenizer
from torchtext.vocab import Vectors
from scipy.special import expit as sigmoid
import os

# colors
colors = {
    'white': '#FFFFFF',
    'text': '#091D58',
    'blue1' : '#063446', #dark blue
    'blue2' : '#0e749b',
    'blue3' : '#15b3f0',
    'blue4' : '#E4F3F9', #light blue
    'yellow1' : '#f0d515'
}

class MovieReviewsModelRunner:
    def __init__(self, model, word_vectors, max_filter_size):
        self.run_model = utils.get_function(model)
        self.vocab = Vectors(word_vectors, cache=os.path.dirname(word_vectors))
        self.max_filter_size = max_filter_size    
        self.tokenizer = get_tokenizer('spacy', 'en_core_web_sm')

    def __call__(self, sentences):
        # ensure the input has a batch axis
        if isinstance(sentences, str):
            sentences = [sentences]

        tokenized_sentences = []
        for sentence in sentences:
            # tokenize and pad to minimum length
            tokens = self.tokenizer(sentence)
            if len(tokens) < self.max_filter_size:
                tokens += ['<pad>'] * (self.max_filter_size - len(tokens))
            
            # numericalize the tokens
            tokens_numerical = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>']
                                for token in tokens]
            tokenized_sentences.append(tokens_numerical)
            
        # run the model, applying a sigmoid because the model outputs logits
        logits = self.run_model(tokenized_sentences)
        pred = np.apply_along_axis(sigmoid, 1, logits)
        
        # output two classes
        positivity = pred[:, 0]
        negativity = 1 - positivity
        return np.transpose([negativity, positivity])

def blank_fig(text=None):
    fig = go.Figure(data=go.Scatter(x=[], y=[]))
    fig.update_layout(
        paper_bgcolor = colors['blue4'],
        plot_bgcolor = colors['blue4'],
        width=300,
        height=300)

    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)

    if text is not None:
        fig.update_layout(
            width=300,
            height=300,
            annotations = [
                        {   
                            "text": text,
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                                "size": 14,
                                "color" : colors['blue1']
                            },
                            "valign": "top",
                            "yanchor": "top",
                            "xanchor": "center",
                            "yshift": 60,
                            "xshift": 10
                        }
                    ]
            )

    return fig

def open_image(path):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    im = np.asarray(im).astype(np.float32)

    if sum(stat.sum)/3 == stat.sum[0]: #check the avg with any element value
        return np.expand_dims(im[:,:,0], axis=2) / 255 #if grayscale
    else:
        return im #else its colour

def parse_contents_image(contents, filename):
    return html.Div([
        html.H5(filename + ' loaded'),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, height = '160 px', width = 'auto')
    ])

def parse_contents_model(contents, filename):
    return html.Div([
        html.H5(filename + ' loaded')
    ])

# For KernelSHAP: fill each pixel with SHAP values
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out

# For LIME: we divided the input data by 256 for the models and LIME needs RGB values
def preprocess_function(image):
    return (image / 256).astype(np.float32)

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