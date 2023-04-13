import numpy as np
import sklearn
from fastdtw import fastdtw
from lime import explanation
from lime import lime_base
from tqdm import tqdm
from dianna import utils
from dianna.utils.maskers import generate_masks
from dianna.utils.maskers import mask_data


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
        # feature_selection="auto",
    ):
        """Initializes Lime explainer for timeseries.

        Args:
            kernel_width (int): Width of the kernel used in LIME explainer.
            verbose (bool): Whether to print progress messages during explanation.
            feature_selection (str): Feature selection method to be used by explainer.
            preprocess_function (callable, optional): Function to preprocess the time series data before passing it to the explainer. Defaults to None.
        """

        def kernel(d):
            """Kernel function used in LIME explainer."""
            return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.explainer = lime_base.LimeBase(kernel, verbose)
        # self.feature_selection = feature_selection
        self.domain_mapper = explanation.DomainMapper()
        self.preprocess_function = preprocess_function
        self._is_multivariate = False

    def explain(
        self,
        model_or_function,
        input_timeseries,
        labels,
        class_names,
        num_features,
        num_samples,
        num_slices,
        batch_size=1,
        mask_type="mean",
        distance_method="cosine",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run the LIME explainer for timeseries.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_timeseries (np.ndarray): Input timeseries data to be explained, the shape must be []
            class_names : Names of classes
            distance_method : Methods for calculating distance
            labels : Labels for different classes
            mask_type : Type of masking
            num_features : Number of features
            num_samples : Number of samples
            num_slices : Number of slices

        Args:
            model_or_function (callable or str): A function that runs the model to be explained or the path to a ONNX model on disk.
            input_timeseries (np.ndarray): The input time series data to be explained, with shape [batch_size, sequence_length, num_features].
            labels (list): The list of labels for different classes.
            class_names (list): The list of class names.
            num_features (int): The number of features to include in the explanation.
            num_samples (int): The number of samples to generate for the LIME explainer.
            num_slices (int): The number of slices to divide the time series data into.
            batch_size (int): The batch size to use for running the model.
            mask_type (str): The type of mask to apply to the time series data. Can be "mean" or "noise".
            distance_method (str): The distance metric to use for LIME. Can be "cosine" or "euclidean".

        Returns:
            explanation: An Explanation object containing the LIME explanations for each class.
        """
        # TODO: p_keep does not exist in LIME. LIME will mask every point, which means the number
        #       of steps masked is 1. We should updating it after adapting maskers function to LIME.
        # wrap up the input model or function using the runner
        runner = utils.get_function(
            model_or_function, preprocess_function=self.preprocess_function
        )
        masks = generate_masks(input_timeseries, num_samples, p_keep=0.9)
        masked = mask_data(input_timeseries, masks, mask_type="mean")
        # generate predictions using the masked data.
        predictions = self._make_predictions(masked, runner, batch_size)
        # need to reshape for the calculation of distance
        _, sequence, n_var = masked.shape
        masked = masked.reshape((-1, sequence * n_var))
        distance = self._calculate_distance(
            input_timeseries, masked, distance_method=distance_method
        )
        exp = explanation.Explanation(
            domain_mapper=self.domain_mapper, class_names=class_names
        )
        # TODO: The current form of explanation follows lime-for-time. Would be good to merge formatting with DIANNA.
        # run the explanation.
        # https://github.com/emanuel-metzenthin/Lime-For-Time/blob/3af530f778ab2593246cefc1e5fdb28fa872dbdf/lime_timeseries.py#L130

        # NOTE: the first instance in masked should be the original data, so it is with the predictions and
        # distance (therefore 1). Check the following link for the explanation
        # https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_base.py#L148
        # expected shape of input
        # masked: [num_samples, channels * num_slices]
        # predictions: [num_samples, labels]
        # distances: [num_samples]
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
        return exp

    def _calculate_distance(self, input_data, masked_data, distance_method="cosine"):
        """Calcuate distance between perturbed data and the original samples.

        Args:
            input_data (np.ndarray): The original time series data.
            masked_data (np.ndarray): The perturbed time series data.
            distance_method (str): The distance metric to use. Defaults to "cosine".
                Supported options are:
                - 'cosine': Computes the cosine similarity between the two vectors.
                - 'euclidean': Computes the Euclidean distance between the two vectors.
                - 'dtw': Uses Dynamic Time Warping to calculate the distance between the two time series.

        Returns:
            np.ndarray: A vector containing the distance between two timeseries.

        Raises:
            ValueError: If the given `distance_method` is not supported.

        Notes:
            - The cosine similarity is a measure of the similarity between two non-zero vectors of an inner
            product space that measures the cosine of the angle between them.
            - The Euclidean distance is the straight-line distance between two points in Euclidean space.
            - Dynamic Time Warping is an algorithm for measuring similarity between two time series sequences
            that may vary in speed or timing.

        """
        support_methods = ["cosine", "euclidean"]
        if distance_method == "dtw":
            distance = self._dtw_distance(input_data, masked_data)
        elif distance_method in support_methods:
            # TODO: implementation for reference
            # https://github.com/emanuel-metzenthin/Lime-For-Time/blob/3af530f778ab2593246cefc1e5fdb28fa872dbdf/lime_timeseries.py#L175
            # should understand why (* 100?) and if it is equivalent to dtw.
            distance = (
                sklearn.metrics.pairwise.pairwise_distances(
                    masked_data, masked_data[0].reshape([1, -1]), metric=distance_method
                ).ravel()
                * 100
            )
        else:
            raise ValueError(
                f"Given method {distance_method} is not supported. Please "
                "choose from 'dtw', 'cosine' and 'euclidean'."
            )

        return distance

    def _dtw_distance(self, input_data, masked_data):
        """Calculate distance based on dynamic time warping.

        Args:
            input_data (np.ndarray): The input time series.
            masked_data (np.ndarray): An array of time series with some segments masked out.

        Returns:
            np.ndarray: DTW distances.
        """
        # implementation for reference
        # https://github.com/TortySivill/LIMESegment/blob/0a276e30f8d259642521407e7d51d07969169432/Utils/explanations.py#L111
        distance = np.asarray(
            [fastdtw(input_data, one_masked_data)[0] for one_masked_data in masked_data]
        )
        return distance

    # TODO: duplication code from rise_timeseries. Need to put it in util.py
    def _make_predictions(self, masked_data, runner, batch_size):
        """Make predictions for the masked data.

        Process the masked_data with the model runner and return the predictions.

        Args:
            masked_data (np.ndarray): An array of masked input data to be processed by the model.
            runner (object): An object that runs the model on the input data and returns the predictions.
            batch_size (int): The number of masked inputs to process in each batch.

        Returns:
            np.ndarray: An array of predictions made by the model on the input data.
        """
        number_of_masks = masked_data.shape[0]
        predictions = []
        for i in tqdm(range(0, number_of_masks, batch_size), desc="Explaining"):
            predictions.append(runner(masked_data[i : i + batch_size]))
        return np.concatenate(predictions)
