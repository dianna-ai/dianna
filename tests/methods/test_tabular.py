import numpy as np
import dianna
from tests.utils import run_model


def assert_tabular_regression_correct_output_shape(method):
    """Runs the explainer class with random data and asserts the output shape."""
    training_data = np.random.random((10, 2))
    input_data = np.random.random(2)
    feature_names = ["feature_1", "feature_2"]
    exp = dianna.explain_tabular(run_model,
                                 input_tabular=input_data,
                                 method=method,
                                 mode='regression',
                                 training_data=training_data,
                                 feature_names=feature_names,
                                 class_names=['class_1'])
    assert len(exp) == len(feature_names)


def assert_tabular_classification_correct_output_shape(explainer_class):
    """Runs the explainer class with random data and asserts the output shape."""
    training_data = np.random.random((10, 2))
    input_data = np.random.random(2)
    feature_names = ["feature_1", "feature_2"]
    explainer = explainer_class(training_data,
                                mode='classification',
                                feature_names=feature_names,
                                class_names=["class_1", "class_2"])
    exp = explainer.explain(
        run_model,
        input_data,
    )
    assert len(exp[0]) == len(feature_names)
