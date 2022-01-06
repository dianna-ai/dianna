from unittest import TestCase

import dianna
import numpy as np
from dianna.methods import shap
from tests.utils import run_model

class ShapOnImages(TestCase):

    def test_shap_segment_image(self):
        input_data = np.random.random((28,28,1))

        explainer = dianna.methods.KernelSHAP()
        # most arguments are chosen by default
        # https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        n_segments=50
        compactness=10.0
        sigma=0
        channel_axis = -1
        image_segments = explainer._segment_image(input_data, n_segments, compactness,
                                                  10, sigma, None, None, True, 0.5, 3,
                                                  False, 1, None, channel_axis)
        # check segments index
        assert (np.array_equal(np.unique(image_segments), np.arange(1, n_segments)))
        # check image shape after segmentation
        assert image_segments.shape == input_data[:,:,0].shape
    
    def test_shap_mask_image(self):
        # check image with channel axis = -1
        input_data = np.random.random((28,28,1))
        explainer = dianna.methods.KernelSHAP()
        n_segments=50
        channel_axis = -1
        segments_slic = explainer._segment_image(input_data, n_segments, 10.0,
                                                 10, 0, None, None, True, 0.5, 3,
                                                 False, 1, None, channel_axis)
        masked_image = explainer._mask_image(np.zeros((1,n_segments)), segments_slic, input_data, 0)
        # check if all points are masked
        assert (np.array_equal(masked_image[0], np.zeros(input_data.shape)))

        # check image with channel axis = 0
        input_data = np.random.random((1,28,28))
        channel_axis = 0
        segments_slic = explainer._segment_image(input_data, n_segments, 10.0,
                                                 10, 0, None, None, True, 0.5, 3,
                                                 False, 1, None, channel_axis)
        masked_image = explainer._mask_image(np.zeros((1,n_segments)), segments_slic, input_data, 0)
        # check if all points are masked
        assert (np.array_equal(masked_image[0], np.zeros(input_data.shape)))
    
    def test_shap_explain_image(self):
        input_data = np.random.random((1, 28, 28, 1))
        