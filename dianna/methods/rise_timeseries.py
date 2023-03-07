import numpy as np
from tqdm import tqdm

from dianna.utils.maskers import mask_time_steps


class RISETimeseries:
    def __init__(self, n_masks=1000, feature_res=8, p_keep=None,
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

    def explain(self, model_or_function, input_timeseries, labels, tokenizer=None,  # pylint: disable=too-many-arguments
                batch_size=100):
        masked = mask_time_steps(input_timeseries, number_of_masks=self.n_masks, p_keep=self.p_keep, mask_type='mean')

        _input_data, runner = self._prepare_input_data_and_model(input_timeseries, model_or_function)

        batch_predictions = []
        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            batch_predictions.append(runner(masked[i:i + batch_size]))
        self.predictions = np.concatenate(batch_predictions)

        sailancy_map = self.predictions.dot(masks)
        return np.zeros([len(labels)] + list(input_timeseries.shape))

    def _prepare_input_data_and_model(self, input_data, model_or_function):
        """Prepares the input data as an xarray with an added batch dimension and creates a preprocessing function."""
        self._set_axis_labels(input_data)
        input_data = utils.to_xarray(input_data, self.axis_labels)
        input_data = input_data.expand_dims('batch', 0)
        input_data, full_preprocess_function = self._prepare_image_data(input_data)
        runner = utils.get_function(model_or_function, preprocess_function=full_preprocess_function)
        return input_data, runner
