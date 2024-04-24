import numpy as np
import pytest
import dianna
from dianna.methods.kernelshap_tabular import KERNELSHAPTabular
from dianna.methods.lime_tabular import LIMETabular
from dianna.methods.rise_tabular import RISETabular
from tests.utils import run_model


@pytest.mark.parametrize('method', [
    'rise',
    'lime',
    'kernelshap',
])
def test_tabular_regression_correct_output_shape(method):
    """Runs the explainer class with random data and asserts the output shape."""
    training_data = np.random.random((10, 2))
    input_data = np.random.random(2)
    feature_names = ["feature_1", "feature_2"]
    exp = dianna.explain_tabular(
        run_model,
        input_tabular=input_data,
        method=method,
        mode='regression',
        training_data=training_data,
        feature_names=feature_names,
    )
    assert len(exp) == len(feature_names)


@pytest.mark.parametrize('explainer_class', [
    RISETabular,
    LIMETabular,
    KERNELSHAPTabular,
])
def test_tabular_classification_correct_output_shape(explainer_class):
    """Runs the explainer class with random data and asserts the output shape."""
    training_data = np.random.random((10, 2))
    input_data = np.random.random(2)
    feature_names = ["feature_1", "feature_2"]
    explainer = explainer_class(
        training_data,
        mode='classification',
        feature_names=feature_names,
    )
    exp = explainer.explain(
        run_model,
        input_data,
        labels=[0],
    )
    assert len(exp[0]) == len(feature_names)


def _pprint(explanations):
    """Pretty prints the explanation for each class while classifying tabular data."""
    print()
    rows = [' '.join([f'{v:>4d}' for v in range(25)])]
    rows += [
        ' '.join([f'{v:.2f}' for v in explanation])
        for explanation in explanations
    ]
    print('\n'.join(rows))


@pytest.mark.parametrize('explainer_class', [
    RISETabular,
    LIMETabular,
    KERNELSHAPTabular,
])
def test_tabular_simple_dummy_model(explainer_class):
    """Tests if the explainer can find the single important feature in otherwise random data."""
    np.random.seed(0)
    num_features = 25
    input_data = np.array(num_features // 2 * [1.0] +
                          (num_features - num_features // 2) * [0.0])
    training_data = np.stack([input_data for _ in range(len(input_data))]).T

    feature_names = [f"feature_{i}" for i in range(num_features)]
    important_feature_i = 2

    def dummy_model(tabular_data):
        """Model with output dependent on a single feature of the first instance."""
        prediction = tabular_data[:, important_feature_i]
        return np.vstack([prediction, -prediction + 1]).T

    explainer = explainer_class(
        training_data,
        mode='classification',
        feature_names=feature_names,
    )
    explanations = explainer.explain(
        dummy_model,
        input_data,
        labels=[0, 1],
    )

    _pprint(explanations)

    assert np.argmax(explanations[0]) == important_feature_i
    assert np.argmin(explanations[1]) == important_feature_i
