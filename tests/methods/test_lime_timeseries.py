from unittest import TestCase
import numpy as np
from dianna.methods.lime_timeseries import LIMETimeseries
from dianna.utils.maskers import generate_time_series_masks
from dianna.utils.maskers import mask_data
from tests.utils import get_dummy_model_function


class LIMEOnTimeseries(TestCase):
    """Suite of LIME tests for the timeseries case."""

    def test_lime_timeseries_correct_output_shape(self):
        """Test the output of explainer."""
        input_data = np.random.random((10, 1))
        num_features = 10
        explainer = LIMETimeseries()
        exp = explainer.explain(
            get_dummy_model_function(n_outputs=2),
            input_data,
            labels=(0, ),
            class_names=("test", ),
            num_features=num_features,
            num_samples=10,
            num_slices=10,
            mask_type="mean",
        )
        assert len(exp[0]) == num_features

    def test_distance_shape(self):
        """Test the shape of returned distance array."""
        dummy_timeseries = np.random.random((50, 1))
        number_of_masks = 50
        masks = generate_time_series_masks(dummy_timeseries.shape,
                                           number_of_masks,
                                           p_keep=0.9)
        masked = mask_data(dummy_timeseries, masks, mask_type="mean")
        explainer = LIMETimeseries()
        distance = explainer._calculate_distance(masked.reshape((-1, 50)),
                                                 distance_method="cosine")
        assert len(distance) == number_of_masks

    def test_euclidean_distance(self):
        """Test the calculation of cosine and euclidean distance."""
        masked_data = np.array([[1, 2, 3, 4, 5], [0, 0, 3, 4, 5]])
        expected_distance = np.array([0.0, 2.236068])
        explainer = LIMETimeseries()
        distance = explainer._calculate_distance(masked_data,
                                                 distance_method="euclidean")
        np.testing.assert_array_almost_equal(distance, expected_distance)

    def test_cosine_distance(self):
        """Test the calculation of cosine and euclidean distance."""
        masked_data = np.array([[1, 2, 3, 4, 5], [0, 0, 3, 4, 5]])
        expected_distance = np.array([0.0, 4.653741])
        explainer = LIMETimeseries()
        distance = explainer._calculate_distance(masked_data,
                                                 distance_method="cosine")
        np.testing.assert_array_almost_equal(distance, expected_distance)

    def test_dtw_distance(self):
        """Test DTW distance."""
        # Create some test data
        masked_data = np.array([[1, 2, 3, 4, 5], [0, 0, 3, 4, 5]])

        # Calculate the expected results
        expected_distance = np.array([0.0, 3.0])

        # Calculate the distance
        explainer = LIMETimeseries()
        distance = explainer._dtw_distance(masked_data)

        # Check that the calculated and expected results are equal
        np.testing.assert_array_almost_equal(distance, expected_distance)
