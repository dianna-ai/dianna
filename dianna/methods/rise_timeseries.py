import numpy as np
from tqdm import tqdm
from dianna import utils
from dianna.utils.maskers import generate_masks
from dianna.utils.maskers import mask_data


def _make_predictions(input_data, runner, batch_size):
    """Process the input_data with the model runner in batches and return the predictions."""
    number_of_masks = input_data.shape[0]
    batch_predictions = []
    for i in tqdm(range(0, number_of_masks, batch_size), desc='Explaining'):
        batch_predictions.append(runner(input_data[i:i + batch_size]))
    return np.concatenate(batch_predictions)


# TODO: Duplicate code from rise.py:
def normalize(saliency, n_masks, p_keep):
    """Normalizes salience by number of masks and keep probability."""
    return saliency / n_masks / p_keep


class RISETimeseries:
    """RISE implementation for timeseries adapted from the image version of RISE."""

    def __init__(self, n_masks=1000, feature_res=8, p_keep=0.5,
                 preprocess_function=None):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of input data to keep in each mask (Default: auto-tune this value).
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None

    def explain(self, model_or_function, input_timeseries, labels, batch_size=100, mask_type='mean'):
        """Runs the RISE explainer on images.

           The model will be called with masked timeseries,
           with a shape defined by `batch_size` and the shape of `input_data`.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_timeseries (np.ndarray): Input timeseries data to be explained
            batch_size (int): Batch size to use for running the model.
            labels (Iterable(int)): Labels to be explained
            mask_type: Masking strategy for masked values. Choose from 'mean' or a callable(input_timeseries)

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        self.masks = generate_masks(input_timeseries, number_of_masks=self.n_masks, p_keep=self.p_keep)
        masked = mask_data(input_timeseries, self.masks, mask_type=mask_type)

        self.predictions = _make_predictions(masked, runner, batch_size)
        n_labels = self.predictions.shape[1]

        saliency = self.predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(n_labels,
                                                                                        *input_timeseries.shape)
        selected_saliency = saliency[labels]
        return normalize(selected_saliency, self.n_masks, self.p_keep)
