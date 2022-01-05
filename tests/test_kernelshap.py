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
        image_segments = explainer._segment_image(input_data, n_segments, compactness,
                                                  10, sigma, None, True, None, True,
                                                  0.5, 3, False, 1, None)
        # check segments index
        assert (np.unique(image_segments) == np.arange(1, n_segments)).all
        # check image shape after segmentation (1st dimension)
        assert image_segments.shape == input_data[:,:,0].shape
    
    
        