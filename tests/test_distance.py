from pathlib import Path
from unittest import TestCase

import numpy as np

import dianna
import dianna.visualization
from tests.utils import run_model


class DistanceForImages(TestCase):
    """Suite of Distance tests for the image case."""

    def test_distance_function_correct_shape(self):
        """Test if the outputs are the correct shape given some data and a model function."""
        input_data = np.random.random((224, 224, 3))
        axis_labels = ['y', 'x', 'channels']
        embedded_reference = np.array([[0.5, 0.5]])

        heatmaps = dianna.explain_image_distance(run_model, input_data, embedded_reference, axis_labels=axis_labels,
                                                 n_masks=5, p_keep=.5)

        assert heatmaps[0].shape == input_data.shape[:2]

    def test_distance_exact_result(self):
        """Regression test. We test for the exact result that was outputted today."""
        np.random.seed(0)
        input_data = np.random.random((64, 64, 3))
        axis_labels = ['y', 'x', 'channels']
        embedded_reference = np.array([[0.5, 0.5]])

        heatmaps = dianna.explain_image_distance(run_model, input_data, embedded_reference, axis_labels=axis_labels,
                                                 n_masks=5, p_keep=.5)

        expected_heatmaps = np.load(Path(__file__).parent / 'test_data/test_distance_exact_result.npy')
        np.testing.assert_allclose(heatmaps, expected_heatmaps, atol=1e-8)
