from typing import Optional
import numpy as np
from dianna import utils
from dianna.utils.maskers import generate_masks
from dianna.utils.maskers import mask_data
from dianna.utils.predict import make_predictions
from dianna.utils.rise_utils import normalize


class RISETimeseries:
    """RISE implementation for timeseries adapted from the image version of RISE."""

    def __init__(
        self,
        n_masks: int = 1000,
        feature_res: int = 8,
        p_keep: float = 0.5,
        preprocess_function: Optional[callable] = None,
        keep_masks: bool = False,
        keep_masked_data: bool = False,
        keep_predictions: bool = False,
    ) -> np.ndarray:
        """RISE initializer.

        Args:
            n_masks: Number of masks to generate.
            feature_res: Resolution of features in masks.
            p_keep: Fraction of input data to keep in each mask (Default: auto-tune this value).
            preprocess_function: Function to preprocess input data with
            keep_masks: keep masks in memory for the user to inspect
            keep_masked_data: keep masked data in memory for the user to inspect
            keep_predictions: keep model predictions in memory for the user to inspect
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.masked = None
        self.predictions = None
        self.keep_masks = keep_masks
        self.keep_masked_data = keep_masked_data
        self.keep_predictions = keep_predictions

    def explain(self,
                model_or_function,
                input_timeseries,
                labels,
                batch_size=100,
                mask_type='mean'):
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
        runner = utils.get_function(
            model_or_function, preprocess_function=self.preprocess_function)

        masks = generate_masks(input_timeseries,
                               number_of_masks=self.n_masks,
                               feature_res=self.feature_res,
                               p_keep=self.p_keep)
        self.masks = masks if self.keep_masks else None
        masked = mask_data(input_timeseries, masks, mask_type=mask_type)
        self.masked = masked if self.keep_masked_data else None
        predictions = make_predictions(masked, runner, batch_size)
        self.predictions = predictions if self.keep_predictions else None
        n_labels = predictions.shape[1]

        saliency = predictions.T.dot(masks.reshape(self.n_masks, -1)).reshape(
            n_labels, *input_timeseries.shape)
        selected_saliency = saliency[labels]
        return normalize(selected_saliency, self.n_masks, self.p_keep)
