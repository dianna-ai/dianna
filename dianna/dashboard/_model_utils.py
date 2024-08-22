from pathlib import Path
import numpy as np
import onnx
import pandas as pd


def load_data(file):
    """Open data from a file and returns it as pandas DataFrame"""
    df = pd.read_csv(file, parse_dates=True)
    # Add index column
    df.insert(0, 'Index', df.index)
    return df


def preprocess_function(image):
    """For LIME: we divided the input data by 256 for the model (binary mnist) and LIME needs RGB values."""
    return (image / 256).astype(np.float32)


def fill_segmentation(values, segmentation):
    """For KernelSHAP: fill each pixel with SHAP values."""
    out = np.zeros(segmentation.shape)
    for i, _ in enumerate(values):
        out[segmentation == i] = values[i]
    return out


def load_model(file):
    onnx_model = onnx.load(file)
    return onnx_model


def load_labels(file):
    if isinstance(file, (str, Path)):
        file = open(file, 'rb')

    labels = [line.decode().rstrip() for line in file.readlines()]
    if labels is None or labels == ['']:
        raise ValueError(labels)
    return labels


def load_training_data(file):
    return np.float32(np.load(file, allow_pickle=False))
