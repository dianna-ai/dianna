"""Test RISE tabular method."""
from dianna.methods.rise_tabular import RISETabular
from tests.methods.test_tabular import assert_tabular_classification_correct_output_shape
from tests.methods.test_tabular import assert_tabular_regression_correct_output_shape
from tests.methods.test_tabular import assert_tabular_simple_dummy_model


def test_rise_tabular_classification_correct_output_shape():
    """Test the output of explainer."""
    assert_tabular_classification_correct_output_shape(RISETabular)


def test_rise_tabular_regression_correct_output_shape():
    """Test the output of explainer."""
    assert_tabular_regression_correct_output_shape('rise')


def test_rise_tabular_find_max_of_simple_model():
    """Test if RISE finds the single important feature."""
    assert_tabular_simple_dummy_model(RISETabular)
