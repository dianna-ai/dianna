import numpy as np
import sklearn
from fastdtw import fastdtw
from lime import explanation
from lime import lime_base
from dianna import utils
from dianna.utils.maskers import generate_time_series_masks
from dianna.utils.maskers import mask_data
from dianna.utils.predict import make_predictions


class LIMETimeseries:
    """LIME implementation for timeseries.

    This implementation is inspired by the paper:
    Validation of XAI explanations for multivariate time series classification in
    the maritime domain. (https://doi.org/10.1016/j.jocs.2021.101539)
    """

    def __init__(
        self,
        kernel_width=25,
        verbose=False,
        preprocess_function=None,
        feature_selection='auto',
        random_state = None
    ):
        """Initializes Lime explainer for timeseries.

        Args:
            kernel_width (int): Width of the kernel used in LIME explainer.
            verbose (bool): Whether to print progress messages during explanation.
            feature_selection (str): Feature selection method to be used by explainer.
            preprocess_function (callable, optional): Function to preprocess the time series data before passing it
                                                      to the explainer. Defaults to None.
            random_state (int or np.RandomState, optional): seed or random state. Unused variable for current ts method
        """

        def kernel(d):
            """Kernel function used in LIME explainer."""
            return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.explainer = lime_base.LimeBase(kernel, verbose)
        self.feature_selection = feature_selection
        self.domain_mapper = explanation.DomainMapper()
        self.preprocess_function = preprocess_function
        self._is_multivariate = False

    def explain(
        self,
        model_or_function,
        input_timeseries,
        labels=(0, ),
        class_names=None,
        num_features=1,
        num_samples=1,
        num_slices=1,
        batch_size=1,
        mask_type='mean',
        distance_method='cosine',
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run the LIME explainer for timeseries.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_timeseries (np.ndarray): The input time series data to be explained, with shape
                                           [batch_size, sequence_length, num_features].
            labels (list): The list of labels for different classes.
            class_names (list): The list of class names.
            num_features (int): The number of features to include in the explanation.
            num_samples (int): The number of samples to generate for the LIME explainer.
            num_slices (int): The number of slices to divide the time series data into.
            batch_size (int): The batch size to use for running the model.
            mask_type (str): The type of mask to apply to the time series data. Can be "mean" or "noise".
            distance_method (str): The distance metric to use for LIME. Can be "cosine" or "euclidean".

        Returns:
            np.ndarray: An array (np.ndarray) containing the LIME explanations for each class.
        """
        # TODO: p_keep does not exist in LIME. LIME will mask every point, which means the number
        #       of steps masked is 1. We should updating it after adapting maskers function to LIME.
        # wrap up the input model or function using the runner
        runner = utils.get_function(
            model_or_function, preprocess_function=self.preprocess_function)
        masks = generate_time_series_masks(input_timeseries,
                                           num_samples,
                                           p_keep=0.1)
        # NOTE: Required by `lime_base` explainer since the first instance must be the original data
        # For more details, check this link
        # https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_base.py#L148
        masks[0, :, :] = 1.0
        masked = mask_data(input_timeseries, masks, mask_type=mask_type)
        # generate predictions using the masked data.
        predictions = make_predictions(masked, runner, batch_size)
        # need to reshape for the calculation of distance
        _, sequence, n_var = masked.shape
        masked = masked.reshape((-1, sequence * n_var))
        distance = self._calculate_distance(masked,
                                            distance_method=distance_method)
        exp = explanation.Explanation(domain_mapper=self.domain_mapper,
                                      class_names=class_names)
        # Expected shape of input:
        # masked[num_samples, channels * num_slices],
        # predictions[num_samples, labels],
        # distances[num_samples]
        for label in labels:
            (
                exp.intercept[int(label)],
                exp.local_exp[int(label)],
                exp.score,
                exp.local_pred,
            ) = self.explainer.explain_instance_with_data(
                masked,
                predictions,
                distance,
                label=label,
                num_features=num_features,
                model_regressor=None,
            )
        # extract scores from lime explainer
        saliency = []
        for i, label in enumerate(labels):
            local_exp = sorted(exp.local_exp[label])
            # shape of local_exp [(index, saliency)]
            selected_saliency = [i[1] for i in local_exp]
            saliency.append(selected_saliency[:])

        return np.concatenate(saliency).reshape(-1, sequence, n_var)

    def _calculate_distance(self, masked_data, distance_method='cosine'):
        """Calcuate distance between perturbed data and the original samples.

        Args:
            masked_data (np.ndarray): The perturbed time series data.
                 *Note: The first instance is the original timeseries
            distance_method (str): The distance metric to use. Defaults to "cosine".
                Supported options are:
                - 'cosine': Computes the cosine similarity between the two vectors.
                - 'euclidean': Computes the Euclidean distance between the two vectors.
                - 'dtw': Uses Dynamic Time Warping to calculate the distance between
                  the two time series.

        Returns:
            np.ndarray: A vector containing the distance between two timeseries.

        Raises:
            ValueError: If the given `distance_method` is not supported.

        Notes:
            - The cosine similarity is a measure of the similarity between two non-zero vectors
            of an inner product space that measures the cosine of the angle between them.
            - The Euclidean distance is the straight-line distance between two points in
              Euclidean space.
            - Dynamic Time Warping is an algorithm for measuring similarity between two time
              series sequences that may vary in speed or timing.
        """
        support_methods = ['cosine', 'euclidean']
        if distance_method == 'dtw':
            distance = self._dtw_distance(masked_data)
        elif distance_method in support_methods:
            distance = (sklearn.metrics.pairwise.pairwise_distances(
                masked_data,
                masked_data[0].reshape([1, -1]),
                metric=distance_method).ravel())
            if distance_method == 'cosine':
                distance *= 100  # make sure it has same scale as other methods
        else:
            raise ValueError(
                f'Given method {distance_method} is not supported. Please '
                "choose from 'dtw', 'cosine' and 'euclidean'.")

        return distance

    def _dtw_distance(self, masked_data):
        """Calculate distance based on dynamic time warping.

        Args:
            masked_data (np.ndarray): An array of time series with some segments masked out.
                *Note: The first instance is the original timeseries

        Returns:
            np.ndarray: DTW distances.
        """
        distance = np.asarray([
            fastdtw(masked_data[0], one_masked_data)[0]
            for one_masked_data in masked_data
        ])
        return distance
