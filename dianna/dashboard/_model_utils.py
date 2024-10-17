from pathlib import Path
import numpy as np
import onnx
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Iterable
import xgboost

from dianna.utils.tokenizers import SpacyTokenizer


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


def features_eulaw(texts: list[str], model_tag="law-ai/InLegalBERT"):
    """Create features for a list of texts."""
    max_length = 512
    tokenizer = AutoTokenizer.from_pretrained(model_tag)
    model = AutoModel.from_pretrained(model_tag)

    def process_batch(batch: Iterable[str]):
        cropped_texts = [text[:max_length] for text in batch]
        encoded_inputs = tokenizer(cropped_texts, padding='longest', truncation=True, max_length=max_length,
                                   return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        last_hidden_states = outputs.last_hidden_state
        sentence_features = last_hidden_states.mean(dim=1)
        return sentence_features

    dataloader = DataLoader(texts, batch_size=1)  # batch size of 1 was quickest for my development
    features = [process_batch(batch) for batch in tqdm(dataloader, desc='Creating features')]
    return np.array(torch.cat(features, dim=0))


def classify_texts_eulaw(texts: list[str], model_path, return_proba: bool = False):
    """Classifies every text in a list of texts using the xgboost model stored in model_path.

    The xgboost model will be loaded and used to classify the texts. The texts however will first be processed by a
    large language model which will do the feature extraction for every text. The classifications of the
    xgboost model will be returned.
    For training the xgboost model, see train_legalbert_xgboost.py.

    Parameters
    ----------
    texts
        A list of strings of which each needs to be classified.
    model_path
        The path to a stored xgboost model
    return_proba
        return the probabilities of the model

    Returns
    -------
        List of classifications, one for every text in the list

    """
    features = features_eulaw(texts)
    model = xgboost.XGBClassifier()
    model.load_model(model_path)

    if return_proba:
        return model.predict_proba(features)
    return model.predict(features)


class StatementClassifierEUlaw():
    def __init__(self, model_path):
        self.tokenizer = SpacyTokenizer(name='en_core_web_sm')
        self.model_path = model_path

    def __call__(self, sentences):
        # ensure the input has a batch axis
        if isinstance(sentences, str):
            sentences = [sentences]

        probs = classify_texts_eulaw(sentences, self.model_path, return_proba=True)

        model_runner = np.transpose([(probs[:, 0]), (1 - probs[:, 0])])

        return model_runner
