"""Unit tests for KernelSHAP image."""
from unittest import TestCase
import numpy as np
from dianna.methods.kernelshap_image import KERNELSHAPImage


class ShapOnImages(TestCase):
    """Suite of Kernelshap tests for the image case."""

    def test_shap_segment_image(self):
        """Test if the segmentation of images are correct given some data."""
        input_data = np.random.random((28, 28, 1))

        explainer = KERNELSHAPImage()
        # most arguments are chosen by default
        # https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        n_segments = 50
        compactness = 10.0
        sigma = 0
        image_segments = explainer._segment_image(
            input_data,
            n_segments,
            compactness,
            sigma,
        )
        # check segments index
        assert np.amax(image_segments) <= n_segments
        # check image shape after segmentation
        assert image_segments.shape == input_data[:, :, 0].shape

    def test_shap_mask_image(self):
        """Test if the images masks are correct given some data."""
        input_data = np.random.random((28, 28, 1))
        explainer = KERNELSHAPImage()
        n_segments = 50
        compactness = 10.0
        sigma = 0
        background = 0
        segments_slic = explainer._segment_image(
            input_data,
            n_segments,
            compactness,
            sigma,
        )
        masked_image = explainer._mask_image(
            np.zeros((1, n_segments)),
            segments_slic,
            input_data,
            background,
        )
        # check if all points are masked
        assert np.array_equal(masked_image[0], np.zeros(input_data.shape))

    def test_shap_explain_image(self):
        """Tests exact expected output given an image and model for Kernelshap."""
        input_data = np.random.random((1, 28, 28))
        onnx_model_path = "./tests/test_data/mnist_model.onnx"
        n_segments = 50
        explainer = KERNELSHAPImage()
        labels = [0]
        shap_values, segments = explainer.explain(
            onnx_model_path,
            input_data,
            labels,
            nsamples=1000,
            background=0,
            n_segments=n_segments,
            compactness=10.0,
            sigma=0,
        )
        # Check if shapeley values match number of segments
        n_segments = np.unique(segments).size
        assert shap_values[0].shape == np.zeros((1, n_segments)).shape
