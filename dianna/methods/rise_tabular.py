"""RISE tabular explainer."""
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
import numpy as np
from dianna import utils
from dianna.utils.maskers import generate_tabular_masks
from dianna.utils.maskers import mask_data_tabular
from dianna.utils.predict import make_predictions
from dianna.utils.rise_utils import normalize


class RISETabular:
    """RISE explainer for tabular data."""

    def __init__(
        self,
        training_data: np.array,
        mode: str = "classification",
        feature_names: List[str] = None,
        categorical_features: List[int] = None,
        n_masks: int = 1000,
        feature_res: int = 8,
        p_keep: float = 0.5,
        preprocess_function: Optional[callable] = None,
        class_names=None,
        keep_masks: bool = False,
        keep_masked: bool = False,
        keep_predictions: bool = False,
    ) -> np.ndarray:
        """RISE initializer.

        Args:
            n_masks: Number of masks to generate.
            feature_res: Resolution of features in masks.
            p_keep: Fraction of input data to keep in each mask (Default: auto-tune this value).
            preprocess_function: Function to preprocess input data with
            categorical_features: list of categorical features
            class_names: Names of the classes
            feature_names: Names of the features
            mode: Either classification of regression
            training_data: Training data used for imputation of masked features
            keep_masks: keep masks in memory for the user to inspect
            keep_masked: keep masked data in memory for the user to inspect
            keep_predictions: keep model predictions in memory for the user to inspect
        """
        self.training_data = training_data
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.masked = None
        self.predictions = None
        self.keep_masks = keep_masks
        self.keep_masked = keep_masked
        self.keep_predictions = keep_predictions

    def explain(
        self,
        model_or_function: Union[str, callable],
        input_tabular: np.array,
        labels: Iterable[int],
        mask_type: Optional[Union[str, callable]] = 'most_frequent',
        batch_size: Optional[int] = 100,
    ) -> np.array:
        """Run the RISE explainer.

        Args:
            model_or_function: The function that runs the model to be explained
                                                 or the path to a ONNX model on disk.
            input_tabular: Data to be explained.
            labels: Indices of classes to be explained.
            num_samples: Number of samples
            mask_type: Imputation strategy for masked features
            batch_size: Number of samples to process by the model per batch

        Returns:
            explanation: An Explanation object containing the LIME explanations for each class.
        """
        # run the explanation.
        runner = utils.get_function(model_or_function)

        masks = np.stack(
            list(
                generate_tabular_masks(input_tabular.shape,
                                       number_of_masks=self.n_masks,
                                       p_keep=self.p_keep)))
        self.masks = masks if self.keep_masks else None

        masked = mask_data_tabular(input_tabular,
                                   masks,
                                   self.training_data,
                                   mask_type=mask_type)
        self.masked = masked if self.keep_masked else None
        predictions = make_predictions(masked, runner, batch_size)
        self.predictions = predictions if self.keep_predictions else None
        n_labels = predictions.shape[1]

        masks_reshaped = masks.reshape(self.n_masks, -1)

        saliency = predictions.T.dot(masks_reshaped).reshape(
            n_labels, *input_tabular.shape)
        selected_saliency = saliency if labels is None else saliency[labels]
        return normalize(selected_saliency, self.n_masks, self.p_keep)
