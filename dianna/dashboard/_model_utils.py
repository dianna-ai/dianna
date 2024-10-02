from pathlib import Path
import numpy as np
import onnx
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file):
    """Open data from a file and returns it as pandas DataFrame."""
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


def load_sunshine(file):
    """Tabular sunshine example.

    Load the csv file in a pandas dataframe and split the data in a train and test set.
    """
    data = load_data(file)

    # Drop unused columns
    X_data = data.drop(columns=['DATE', 'MONTH', 'Index'])[:-1]
    y_data = data.loc[1:]["BASEL_sunshine"]

    # Split the data
    X_train, X_holdout, _, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
    _, X_test, _, _ = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=0)
    X_test = X_test.reset_index(drop=True)
    X_test.insert(0, 'Index', X_test.index)

    return X_train.to_numpy(dtype=np.float32), X_test

def load_penguins(penguins):
    """Prep the data for the penguin model example as per ntoebook."""
    # Remove categorial columns and NaN values
    penguins_filtered = penguins.drop(columns=['island', 'sex']).dropna()


    # Extract inputs and target
    input_features = penguins_filtered.drop(columns=['species'])
    target = pd.get_dummies(penguins_filtered['species'])

    X_train, X_test, _, _ = train_test_split(input_features, target, test_size=0.2,
                                                    random_state=0, shuffle=True, stratify=target)

    X_test = X_test.reset_index(drop=True)
    X_test.insert(0, 'Index', X_test.index)

    return X_train.to_numpy(dtype=np.float32), X_test
