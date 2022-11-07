import os
import warnings
import numpy as np
import plotly.graph_objects as go
from keras import backend as K
from keras import utils as keras_utils
from PIL import Image
from PIL import ImageStat
from scipy.special import expit as sigmoid
# keras model and preprocessing tools
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from torchtext.data import get_tokenizer
from torchtext.vocab import Vectors
import dianna
from dianna import utils
from dianna.utils import move_axis
from dianna.utils import to_xarray


warnings.filterwarnings('ignore') # disable warnings relateds to versions of tf


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


class Model_imagenet():
    def __init__(self):
        K.set_learning_phase(0)
        self.model = ResNet50()
        self.input_size = (224, 224)
        
    def run_on_batch(self, x):
        return self.model.predict(x)


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


def open_image(path):
    """Open an image from a path and returns it as a numpy array."""
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    im = np.asarray(im).astype(np.float32)

    if sum(stat.sum)/3 == stat.sum[0]:  # check the avg with any element value
        return np.expand_dims(im[:, :, 0], axis=2) / 255, im  # if grayscale
    else: # else it's colour, reshape to 224x224x3 for resnet
        img_norm, img = preprocess_img_rise(path)
        return img_norm, img


def preprocess_img_rise(path):
    '''reshape figure to 224,224 and get colour channel at position 0.
    Also: for resnet preprocessing: normalize the data. This works specifically for ImageNet'''
    img = keras_utils.load_img(path, target_size=(224,224))
    img_data = keras_utils.img_to_array(img)
    img_data = preprocess_input(img_data)
    if img_data.shape[0] != 3:
        # Colour channel is not in position 0; reshape the data
        xarray = to_xarray(img_data, {0: 'height', 1: 'width', 2: 'channels'}) 
        reshaped_data = move_axis(xarray, 'channels', 0)
        img_data = np.array(reshaped_data)
    # definitions for normalisation (for ImageNet)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the values by 255 ([0,1]), and normalize 
         # using mean and standard deviation from values above
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data, img


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



def imagenet_class_name(idx):
    """Returns label of class index."""
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]
