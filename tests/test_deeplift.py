from unittest import TestCase
import numpy as np
import torch
import dianna
import dianna.visualization
from dianna.methods import DeepLift
from tests.utils import load_torch_model

class DeepLiftOnImages(TestCase):

    def test_deeplift_function(self):
        np.random.seed(42)
        # shape is batch, channel, y, x, 
        input_data = np.random.random((1,1,28,28))
        explainer = DeepLift()
        path_to_model = 'tests/test_data/mnistnet_training_checkpoint.pt'
        heatmap = explainer.explain_image(load_torch_model(path_to_model),
                                          input_data,
                                          baselines=input_data*0,
                                          label=1)
        assert heatmap.shape == input_data.shape