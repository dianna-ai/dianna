from unittest import TestCase
import pytest
import dianna
from tests.test_onnx_runner import generate_data
import numpy as np

class initialize_method(TestCase):

    def test_lime_image_correct_kwargs(self):
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        axis_labels = ('channels', 'y', 'x')
        labels = [1]

        dianna.explain_image(model_filename,
                                input_data,
                                method='LIME',
                                labels=labels,
                                kernel=None,
                                kernel_width=25,
                                verbose=False,
                                feature_selection='auto',
                                random_state=None,
                                axis_labels=axis_labels,
                                preprocess_function=None,
                                top_labels=None,
                                num_features=10,
                                num_samples=5000,
                                return_masks=True,
                                positive_only=False,
                                hide_rest=True,
                                )
        
    def test_lime_image_extra_kwarg(self):
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        axis_labels = ('channels', 'y', 'x')
        labels = [1]

        with self.assertWarns(Warning):
            dianna.explain_image(model_filename,
                                    input_data,
                                    method='LIME',
                                    labels=labels,
                                    kernel=None,
                                    kernel_width=25,
                                    verbose=False,
                                    feature_selection='auto',
                                    random_state=None,
                                    axis_labels=axis_labels,
                                    preprocess_function=None,
                                    top_labels=None,
                                    num_features=10,
                                    num_samples=5000,
                                    return_masks=True,
                                    positive_only=False,
                                    hide_rest=True,
                                    extra_kwarg=None
                                    )

