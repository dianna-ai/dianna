"""Unit tests for RISE image."""
from unittest import TestCase
import numpy as np
import dianna
from dianna.methods.rise_image import RISEImage
from dianna.utils import get_function
from tests.methods.test_onnx_runner import generate_data
from tests.utils import get_dummy_model_function
from tests.utils import get_mnist_1_data


class RiseOnImages(TestCase):
    """Suite of RISE tests for the image case."""

    @staticmethod
    def test_rise_function():
        """Test if rise runs and outputs the correct shape given some data and a model function."""
        input_data = np.random.random((224, 224, 3))
        axis_labels = ["y", "x", "channels"]
        labels = [1]
        heatmaps_expected = np.load(
            "tests/test_data/heatmap_rise_function.npy")

        heatmaps = dianna.explain_image(
            get_dummy_model_function(n_outputs=2),
            input_data,
            "RISE",
            labels,
            axis_labels=axis_labels,
            n_masks=200,
            p_keep=0.5,
        )

        assert heatmaps[0].shape == input_data.shape[:2]
        assert np.allclose(heatmaps, heatmaps_expected, atol=1e-5)

    @staticmethod
    def test_rise_filename():
        """Test if rise runs and outputs the correct shape given some data and a model file."""
        model_filename = "tests/test_data/mnist_model.onnx"
        input_data = generate_data(batch_size=1).astype(np.float32)[0]
        heatmaps_expected = np.load(
            "tests/test_data/heatmap_rise_filename.npy")
        labels = [1]

        heatmaps = dianna.explain_image(model_filename,
                                        input_data,
                                        "RISE",
                                        labels,
                                        n_masks=200,
                                        p_keep=0.5)

        assert heatmaps[0].shape == input_data.shape[1:]
        print(heatmaps_expected.shape)
        assert np.allclose(heatmaps, heatmaps_expected, atol=1e-5)

    @staticmethod
    def test_rise_determine_p_keep_for_images():
        """Tests exact expected p_keep given an image and model."""
        np.random.seed(0)
        expected_p_exact_keep = 0.4
        model_filename = "tests/test_data/mnist_model.onnx"
        data = get_mnist_1_data().astype(np.float32)

        p_keep = RISEImage()._determine_p_keep(data,
                                               get_function(model_filename))

        assert np.isclose(p_keep, expected_p_exact_keep)
