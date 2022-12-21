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


def export_cube_files(feature_name, feature_values, grid, export_path):
    """Create a cube frile from the np array of a given feature

    Args:
        feature_name (str): name of the feature
        feature_values (np.ndarray): value sof the feature
        grid (dict): {'x':np.ndarray, 'y': np.ndarray, 'z':np.ndarray}
        export_path (str): folder where to create the cube file

    Returns:
        str: name of the cube file
    """


    bohr2ang = 0.52918

    # individual axis of the grid
    x,y,z = grid['x'], grid['y'], grid['z']

    # extract grid_info
    npts = np.array([len(x),len(y),len(z)])
    res = np.array([x[1]-x[0],y[1]-y[0],z[1]-z[0]])

    # the cuve file is apparently give in bohr
    xmin,ymin,zmin = np.min(x)/bohr2ang,np.min(y)/bohr2ang,np.min(z)/bohr2ang
    scale_res = res/bohr2ang

    fname = os.path.join(export_path, '%s.cube' %(feature_name)) 
    if not os.path.isfile(fname):
        f = open(fname,'w')
        f.write('CUBE FILE\n')
        f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")

        f.write("%5i %11.6f %11.6f %11.6f\n" %  (1,xmin,ymin,zmin))
        f.write("%5i %11.6f %11.6f %11.6f\n" %  (npts[0],scale_res[0],0,0))
        f.write("%5i %11.6f %11.6f %11.6f\n" %  (npts[1],0,scale_res[1],0))
        f.write("%5i %11.6f %11.6f %11.6f\n" %  (npts[2],0,0,scale_res[2]))


        # the cube file require 1 atom
        f.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (0,0,0,0,0))

        last_char_check = True
        for i in range(npts[0]):
            for j in range(npts[1]):
                for k in range(npts[2]):
                    f.write(" %11.5e" % feature_values[i,j,k])
                    last_char_check = True
                    if k % 6 == 5:
                        f.write("\n")
                        last_char_check = False
                if last_char_check:
                    f.write("\n")
        f.close()
    return fname

def get_feature(molgrp):
    """extract the feature of a h5 group

    Args:
        molgrp (h5.Group): Group containing the data of a given molecule

    Returns:
        dict: {name (str): values (np.ndarray)}
    """

    from deeprank.tools import sparse

    nx = len(molgrp['grid_points/x'])
    ny = len(molgrp['grid_points/y'])
    nz = len(molgrp['grid_points/z'])   
    shape = (nx,ny,nz)

    mapgrp = molgrp['mapped_features']
    data_dict = {}

    # loop through all the features
    for data_name in mapgrp.keys():

        # create a dict of the feature {name : value}
        featgrp = mapgrp[data_name]

        for ff in featgrp.keys():
            subgrp = featgrp[ff]
            if not subgrp.attrs['sparse']:
                data_dict[ff] =  subgrp['value'][()]
            else:
                spg = sparse.FLANgrid(sparse=True, index=subgrp['index'][()], 
                                      value=subgrp['value'][()], shape=shape)
                data_dict[ff] =  spg.to_dense()

    return data_dict

def open_deeprank_hdf5(path):
    """pen a hdf5 file generated by deeprank and extract the data of the first molecule

    Args:
        path (str): hdf5 filename

    Returns:
        (str, list(str), dict, dict): name, pdb, grid points, {str: np.ndarray}
    """
    with h5py.File(path,'r') as f5:
        mol_name = list(f5.keys())[0]
        mol = f5[mol_name]
        mol_complex = mol['complex'][()]
        grid = {'x': mol['grid_points']['x'][()], 
                'y': mol['grid_points']['y'][()], 
                'z': mol['grid_points']['z'][()]}
        feature_dict = get_feature(mol)

    return mol_name, mol_complex, grid, feature_dict

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
