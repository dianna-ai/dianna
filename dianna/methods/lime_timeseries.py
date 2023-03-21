#from lime import explanation
from lime import lime_base
import sklearn
from dianna.utils.maskers import generate_masks
from dianna.utils.maskers import mask_data
import fastdtw
import numpy as np


class LimeTimeseries:
    """LIME implementation for timeseries.
    
    This implementation is inspired by the paper:
    Validation of XAI explanations for multivariate time series classification in
    the maritime domain. (https://doi.org/10.1016/j.jocs.2021.101539)
    """
    def __init__(self):
        """Initializes Lime explainer for timeseries."""
        pass

    def explain(self, 
                model_or_function, 
                input_data,
                labels,
                num_features,
                num_samples,
                num_slices,
                mask_type='mean',
                method='cosine'
                ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run the LIME explainer for timeseries.
        """
        # TODO: p_keep does not exist in LIME, we should remove it after adapting
        #       maskers function to LIME.
        masks = generate_masks(input_data, num_samples, p_keep=0.9)
        masked = mask_data(input_data, masks, mask_type='mean')
        distance = self._calculate_distance(input_data, masked, method=method)
        # implementation for reference
        # https://github.com/emanuel-metzenthin/Lime-For-Time/blob/3af530f778ab2593246cefc1e5fdb28fa872dbdf/lime_timeseries.py#L130
        # TODO: scores =  lime_base.explain_instance_with_data()
        
        return distance

    def _calculate_distance(self, input_data, masked_data, method="cosine"):
        """Calcuate distance between perturbed data and the original samples."""
        support_methods = ["cosine", "euclidean"]
        if method == "dtw":
            distance = self._dtw_distance(input_data, masked_data)
        elif method in support_methods:
            # TODO: implementation for reference
            # https://github.com/emanuel-metzenthin/Lime-For-Time/blob/3af530f778ab2593246cefc1e5fdb28fa872dbdf/lime_timeseries.py#L175
            # should understand why (* 100?) and if it is equivalent to dtw.
            distance = sklearn.metrics.pairwise.pairwise_distances(
                        masked_data, masked_data[0].reshape([1, -1]),
                        metric = method).ravel() * 100
        else:
            raise ValueError(f"Given method {method} is not supported. Please "
                             "choose from 'dtw', 'cosine' and 'euclidean'.")

        return distance

    def _dtw_distance(self, input_data, masked_data):
        """Calculate distance based on dynamic time warping."""
        # implementation for reference
        # https://github.com/TortySivill/LIMESegment/blob/0a276e30f8d259642521407e7d51d07969169432/Utils/explanations.py#L111
        distance =  np.asarray(fastdtw(input_data, one_masked_data)[0] for one_masked_data in masked_data)
        return distance
