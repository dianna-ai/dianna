from lime import lime_base
from lime import explanation
import sklearn
from dianna import utils
from dianna.utils.maskers import generate_masks
from dianna.utils.maskers import mask_data
from fastdtw import fastdtw
import numpy as np
from dianna import utils
from tqdm import tqdm


class LIMETimeseries:
    """LIME implementation for timeseries.

    This implementation is inspired by the paper:
    Validation of XAI explanations for multivariate time series classification in
    the maritime domain. (https://doi.org/10.1016/j.jocs.2021.101539)
    """
    def __init__(self,
                 kernel_width=25,
                 verbose=False,
                 feature_selection='auto',
                 preprocess_function=None,
                 ):
        """Initializes Lime explainer for timeseries."""
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.explainer = lime_base.LimeBase(kernel, verbose)
        self.feature_selection = feature_selection
        self.domain_mapper = explanation.DomainMapper()
        self.preprocess_function = preprocess_function
        self._is_multivariate = False

    def explain(self,
                model_or_function,
                input_timeseries,
                labels,
                class_names,
                num_features,
                num_samples,
                num_slices,
                mask_type='mean',
                distance_method='cosine'
                ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run the LIME explainer for timeseries.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_timeseries (np.ndarray): Input timeseries data to be explained, the shape must be []
        """
        # TODO: p_keep does not exist in LIME. LIME will mask every point, which means the number
        #       of steps masked is 1. We should updating it after adapting maskers function to LIME.
        # TODO: support batch processing
        if input_timeseries.ndim > 2:
            raise ValueError("LIME for timeseries only supports input timeseries with shape"
                             "[timeseries, variables]")
        elif input_timeseries.ndim > 1:
            self._is_multivariate = True
        else:
            pass
        # wrap up the input model or function using the runner
        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        masks = generate_masks(input_timeseries, num_samples, p_keep=0.9)
        masked = mask_data(input_timeseries, masks, mask_type='mean')
        # generate predictions using the masked data.
        predictions = self._make_predictions(masked, runner, num_samples)
        if self._is_multivariate:
            _, sequence, n_var = masked.shape
            masked = masked.reshape((-1, sequence * n_var))
        distance = self._calculate_distance(input_timeseries, masked, distance_method=distance_method)
        exp = explanation.Explanation(domain_mapper = self.domain_mapper, class_names = class_names)
        # TODO: The current form of explanation follows lime-for-time. Would be good to merge formatting with DIANNA.
        # run the explanation.
        # https://github.com/emanuel-metzenthin/Lime-For-Time/blob/3af530f778ab2593246cefc1e5fdb28fa872dbdf/lime_timeseries.py#L130

        # NOTE: the first instance in masked should be the original data, so it is with the predictions and distance (therefore 1)
        # check the following link for the explanation
        # https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_base.py#L148
        # expected shape of input
        # masked: [num_samples, channels * num_slices]
        # predictions: [num_samples, labels]
        # distances: [num_samples]
        for label in labels:
            (exp.intercept[int(label)],
             exp.local_exp[int(label)],
             exp.score,
             exp.local_pred) = self.explainer.explain_instance_with_data(masked,
                                                      predictions,
                                                      distance,
                                                      label=label,
                                                      num_features=num_features,
                                                      model_regressor = None,
                                                      )
        return exp

    def _calculate_distance(self, input_data, masked_data, distance_method="cosine"):
        """Calcuate distance between perturbed data and the original samples."""
        support_methods = ["cosine", "euclidean"]
        if distance_method == "dtw":
            distance = self._dtw_distance(input_data, masked_data)
        elif distance_method in support_methods:
            # TODO: implementation for reference
            # https://github.com/emanuel-metzenthin/Lime-For-Time/blob/3af530f778ab2593246cefc1e5fdb28fa872dbdf/lime_timeseries.py#L175
            # should understand why (* 100?) and if it is equivalent to dtw.
            distance = sklearn.metrics.pairwise.pairwise_distances(
                        masked_data, masked_data[0].reshape([1, -1]),
                        metric = distance_method).ravel() * 100
        else:
            raise ValueError(f"Given method {distance_method} is not supported. Please "
                             "choose from 'dtw', 'cosine' and 'euclidean'.")

        return distance

    def _dtw_distance(self, input_data, masked_data):
        """Calculate distance based on dynamic time warping."""
        # implementation for reference
        # https://github.com/TortySivill/LIMESegment/blob/0a276e30f8d259642521407e7d51d07969169432/Utils/explanations.py#L111
        distance = np.asarray([fastdtw(input_data, one_masked_data)[0] for one_masked_data in masked_data])
        return distance

    def _make_predictions(self, masked_data, runner, num_samples):
        """Process the masked_data with the model runner and return the predictions."""
        if not self._is_multivariate:
            return runner(masked_data)
        predictions = []
        for i in tqdm(range(0, num_samples), desc='Explaining'):
            predictions.append(runner(masked_data[i]))
        return np.asarray(predictions)
