from unittest import TestCase
import numpy as np
import pytest

from dianna.methods.lime_timeseries import LimeTimeseries
from tests.utils import run_model
from dianna.utils.maskers import generate_masks
from dianna.utils.maskers import mask_data


class LIMEOnTimeseries(TestCase):
    """Suite of LIME tests for the timeseries case."""

    def test_lime_timeseries_univar(self):
        dummy_timeseries_univar = np.random.random((50))
        explainer = LimeTimeseries()
        with pytest.raises(NotImplementedError):
            explainer.explain(run_model,
                              dummy_timeseries_univar,
                              labels=(1,),
                              num_features=5,
                              num_samples=500,
                              num_slices=10,
                              mask_type='mean'
                              )

    def test_cosine_distance(self):
        dummy_timeseries_univar = np.random.random((50))
        number_of_masks = 500
        masks = generate_masks(dummy_timeseries_univar,
                               number_of_masks,
                               p_keep=0.9)
        masked = mask_data(dummy_timeseries_univar,
                           masks,
                           mask_type='mean')
        explainer = LimeTimeseries()
        distance = explainer._calculate_distance(dummy_timeseries_univar,
                                                 masked, method="cosine")
        assert len(distance) == number_of_masks

    def test_dtw_distance(self):
        dummy_timeseries_univar = np.random.random((50))
        masks = generate_masks(dummy_timeseries_univar,
                               number_of_masks=500,
                               p_keep=0.9)
        masked = mask_data(dummy_timeseries_univar,
                           masks,
                           mask_type='mean')
        explainer = LimeTimeseries()
        with pytest.raises(NotImplementedError):
            explainer._dtw_distance(dummy_timeseries_univar, masked)
