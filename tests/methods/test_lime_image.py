"""Unit tests for LIME image."""
from unittest import TestCase
import numpy as np
import dianna
from dianna.methods.lime_image import LIMEImage
from tests.methods.test_onnx_runner import generate_data
from tests.utils import get_dummy_model_function


class LimeOnImages(TestCase):
    """Suite of Lime tests for the image case."""

    @staticmethod
    def test_lime_function():
        """Test if lime runs and outputs are correct given some data and a model function."""
        input_data = np.random.random((224, 224, 3))
        heatmap_expected = np.load('tests/test_data/heatmap_lime_function.npy')
        labels = [1]

        explainer = LIMEImage(random_state=42)
        heatmap = explainer.explain(get_dummy_model_function(n_outputs=2),
                                    input_data,
                                    labels,
                                    num_samples=100)

        assert heatmap[0].shape == input_data.shape[:2]
        assert np.allclose(heatmap, heatmap_expected, atol=1e-5)

    @staticmethod
    def test_lime_filename():
        """Test if lime runs and outputs are correct given some data and a model file."""
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        axis_labels = ('channels', 'y', 'x')
        labels = [1]

        heatmap = dianna.explain_image(model_filename,
                                       input_data,
                                       method='LIME',
                                       labels=labels,
                                       random_state=42,
                                       axis_labels=axis_labels)

        heatmap_expected = np.load('tests/test_data/heatmap_lime_filename.npy')
        assert heatmap[0].shape == input_data[0].shape
        assert np.allclose(heatmap, heatmap_expected, atol=1e-5)

    @staticmethod
    def test_lime_values():
        """Test if get_explanation_values function works correctly."""
        input_data = np.random.random((224, 224, 3))
        heatmap_expected = np.load('tests/test_data/heatmap_lime_values.npy')
        labels = [1]

        explainer = LIMEImage(random_state=42)
        heatmap = explainer.explain(get_dummy_model_function(n_outputs=2),
                                    input_data,
                                    labels,
                                    return_masks=False,
                                    num_samples=100)

        assert heatmap[0].shape == input_data.shape[:2]
        assert np.allclose(heatmap, heatmap_expected, atol=1e-5)

    def setUp(self) -> None:
        """Set seed."""
        np.random.seed(42)
