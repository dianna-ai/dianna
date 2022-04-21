from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.keras import backend as K
from skimage.transform import resize
from requests import get
import os
from urllib.parse import urlparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import os
import random
import numpy as np
import tensorflow as tf
from keras import backend as K


class Model():
    def __init__(self):
        K.set_learning_phase(0)
        self.model = ResNet50()
        self.input_size = (224, 224)
        
    def run_on_batch(self, x):
        return self.model.predict(x)


def load_img(path, target_size):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def generate_masks(N, s, p1, model_input_size):
    cell_size = np.ceil(np.array(model_input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *model_input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model_input_size[0], y:y + model_input_size[1]]
    masks = masks.reshape(-1, *model_input_size, 1)
    return masks


def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]


def download(url):
    filename = os.path.basename(urlparse(url).path)
    os.system(f"wget {url}")
    return filename


def plot_explainer(image, saliency, ax=None, vmin=None, vmax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    ax.set_title('Explanation')
    ax.axis('off')
    ax.imshow(image)
    im = ax.imshow(saliency, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    return fig


# from https://stackoverflow.com/a/52897216/1199693
def set_all_the_seeds(seed_value=0):
    os.environ['PYTHONHASHSEED']=str(seed_value)

    random.seed(seed_value)

    np.random.seed(seed_value)

    tf.random.set_seed(seed_value)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)