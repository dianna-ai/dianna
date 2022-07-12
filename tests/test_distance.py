from unittest import TestCase

import dianna
import dianna.visualization
import numpy as np
from dianna.methods.distance import DistanceExplainer
from dianna.utils import get_function
from tests.utils import ModelRunner, run_model, get_mnist_1_data

from .test_onnx_runner import generate_data


class DistanceForImages(TestCase):
    """Suite of RISE tests for the image case."""
    def test_distance_function_correct_shape(self):
        """Test if the outputs are the correct shape given some data and a model function."""
        input_data = np.random.random((224, 224, 3))
        axis_labels = ['y', 'x', 'channels']
        embedded_reference = np.array([[0.5, 0.5]])

        heatmaps = dianna.explain_image_distance(run_model, input_data, embedded_reference, method="RISE", axis_labels=axis_labels, n_masks=5, p_keep=.5)

        assert heatmaps[0].shape == input_data.shape[:2]
