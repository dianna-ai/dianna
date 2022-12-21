import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageStat
from dianna import utils
from torchtext.data import get_tokenizer
from torchtext.vocab import Vectors
from scipy.special import expit as sigmoid
import os
import h5py 

# colors
colors = {
    'white': '#FFFFFF',
    'text': '#091D58',
    'blue1': '#063446',  # dark blue
    'blue2': '#0e749b',
    'blue3': '#15b3f0',
    'blue4': '#E4F3F9',  # light blue
    'yellow1': '#f0d515'
}


class MovieReviewsModelRunner:
    """Creates runner for movie review model."""
    def __init__(self, model, word_vectors, max_filter_size):
        """Initializes the class."""
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
    """Creates a blank figure."""
    fig = go.Figure(data=go.Scatter(x=[], y=[]))
    fig.update_layout(
        paper_bgcolor=colors['blue4'],
        plot_bgcolor=colors['blue4'],
        width=300,
        height=300)

    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    if text is not None:
        fig.update_layout(
            width=300,
            height=300,
            annotations=[
                        {
                            "text": text,
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                                "size": 14,
                                "color": colors['blue1']
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


def open_deeprank_hdf5(path):
    with h5py.File(path,'r') as f5:
        mol_name = list(f5.keys())[0]
        mol_complex = f5[mol_name]['complex'][()]
    return mol_name, mol_complex

def open_image(path):
    """Open an image from a path and returns it as a numpy array."""
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    im = np.asarray(im).astype(np.float32)

    if sum(stat.sum)/3 == stat.sum[0]:  # check the avg with any element value
        return np.expand_dims(im[:, :, 0], axis=2) / 255  # if grayscale

    return im  # else it's colour


def fill_segmentation(values, segmentation):
    """For KernelSHAP: fill each pixel with SHAP values."""
    out = np.zeros(segmentation.shape)
    for i, _ in enumerate(values):
        out[segmentation == i] = values[i]
    return out


def preprocess_function(image):
    """For LIME: we divided the input data by 256 for the model (binary mnist) and LIME needs RGB values."""
    return (image / 256).astype(np.float32)

def _create_html(input_tokens, explanation, max_opacity):
    """Creates text explaination map using html format."""
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
    """Defines how to highlight words according to importance."""
    opacity = max_opacity * abs(importance) / max_importance
    if importance > 0:
        color = f'rgba(255, 0, 0, {opacity:.2f})'
    else:
        color = f'rgba(0, 0, 255, {opacity:2f})'
    highlighted_word = f'<span style="background:{color}">{word}</span>'
    return highlighted_word
