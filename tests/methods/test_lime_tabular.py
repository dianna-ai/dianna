"""Test LIME tabular method."""
from dianna.methods.lime_tabular import LIMETabular
from tests.methods.test_tabular import assert_tabular_classification_correct_output_shape
from tests.methods.test_tabular import assert_tabular_regression_correct_output_shape


def test_lime_tabular_classification_correct_output_shape(self):
    """Test the output of explainer."""
    assert_tabular_classification_correct_output_shape(LIMETabular)


def test_lime_tabular_regression_correct_output_shape(self):
    """Test the output of explainer."""
    assert_tabular_regression_correct_output_shape('lime')
