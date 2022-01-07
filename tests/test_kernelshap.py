from unittest import TestCase

import dianna
import numpy as np
from dianna.methods import shap
from tests.utils import run_model


class ShapOnImages(TestCase):
    def test_shap_segment_image(self):
        input_data = np.random.random((28, 28, 1))

        explainer = dianna.methods.KernelSHAP()
        # most arguments are chosen by default
        # https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        n_segments = 50
        compactness = 10.0
        sigma = 0
        image_segments = explainer._segment_image(
            input_data,
            n_segments,
            compactness,
            10,
            sigma,
            None,
            None,
            True,
            0.5,
            3,
            False,
            1,
            None,
            channel_axis_first=False,
        )
        # check segments index
        assert np.amax(image_segments) <= n_segments
        # check image shape after segmentation
        assert image_segments.shape == input_data[:, :, 0].shape

    def test_shap_mask_image(self):
        # check image with channel axis = -1
        input_data = np.random.random((28, 28, 1))
        explainer = dianna.methods.KernelSHAP()
        n_segments = 50
        segments_slic = explainer._segment_image(
            input_data,
            n_segments,
            10.0,
            10,
            0,
            None,
            None,
            True,
            0.5,
            3,
            False,
            1,
            None,
            False,
        )
        masked_image = explainer._mask_image(
            np.zeros((1, n_segments)), segments_slic, input_data, 0, False
        )
        # check if all points are masked
        assert np.array_equal(masked_image[0], np.zeros(input_data.shape))

        # check image with channel axis = 0
        input_data = np.random.random((1, 28, 28))
        segments_slic = explainer._segment_image(
            input_data,
            n_segments,
            10.0,
            10,
            0,
            None,
            None,
            True,
            0.5,
            3,
            False,
            1,
            None,
            True,
        )
        masked_image = explainer._mask_image(
            np.zeros((1, n_segments)), segments_slic, input_data, 0, True
        )
        # check if all points are masked
        assert np.array_equal(masked_image[0], np.zeros(input_data.shape))

    def test_shap_explain_image(self):
        input_data = np.random.random((1, 28, 28, 1))

