"""Test LIME tabular method."""
from unittest import TestCase
import numpy as np
import dianna
from dianna.methods.kernelshap_tabular import KERNELSHAPTabular
from tests.utils import run_model


class LIMEOnTabular(TestCase):
    """Suite of LIME tests for the tabular case."""

    def test_shap_tabular_classification_correct_output_shape(self):
        """Test whether the output of explainer has the correct shape."""
        training_data = np.random.random((10, 2))
        input_data = np.random.random(2)
        feature_names = ["feature_1", "feature_2"]
        explainer = KERNELSHAPTabular(training_data,
                                      mode ='classification',
                                      feature_names=feature_names,)
        exp = explainer.explain(
            run_model,
            input_data,
        )
        assert len(exp[0]) == len(feature_names)

    def test_shap_tabular_regression_correct_output_shape(self):
        """Test whether the output of explainer has the correct length."""
        training_data = np.random.random((10, 2))
        input_data = np.random.random(2)
        feature_names = ["feature_1", "feature_2"]
        exp = dianna.explain_tabular(run_model, input_tabular=input_data, method='kernelshap',
                                     mode ='regression', training_data = training_data,
                                     training_data_kmeans = 2, feature_names=feature_names)

        assert len(exp) == len(feature_names)
