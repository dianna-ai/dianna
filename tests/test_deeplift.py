from unittest import TestCase
import numpy as np
import dianna
import dianna.visualization
from dianna.methods import DeepLift

class DeepLiftOnImages(TestCase):

    def test_lime_function(self):
        np.random.seed(42)
        # shape is batch, y, x, channel
        input_data = np.random.random((1, 224, 224, 3))       
        explainer = DeepLift()
        heatmap = explainer.explain_image(run_model, input_data, num_samples=100)