"""Test LIME tabular method."""
from unittest import TestCase
import numpy as np
import dianna
from dianna.methods.lime_tabular import LIMETabular
from tests.utils import run_model


class LIMEOnTabular(TestCase):
    """Suite of LIME tests for the tabular case."""

    def test_lime_tabular_classification_correct_output_shape(self):
        """Test the output of explainer."""
        training_data = np.random.random((10, 2))
        input_data = np.random.random(2)
        feature_names = ["feature_1", "feature_2"]
        explainer = LIMETabular(training_data,
                                mode ='classification',
                                feature_names=feature_names,
                                class_names = ["class_1", "class_2"])
        exp = explainer.explain(
            run_model,
            input_data,
        )
        assert len(exp[0]) == len(feature_names)

    def test_lime_tabular_regression_correct_output_shape(self):
        """Test the output of explainer."""
        training_data = np.random.random((10, 2))
        input_data = np.random.random(2)
        feature_names = ["feature_1", "feature_2"]
        exp = dianna.explain_tabular(run_model, input_tabular=input_data, method='lime',
                             mode ='regression', training_data = training_data,
                             feature_names=feature_names, class_names=['class_1'])

        assert len(exp) == len(feature_names)
