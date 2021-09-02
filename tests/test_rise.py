import numpy as np
import dianna
import dianna.visualization
from .test_onnx_runner import generate_data


def run_model(input_data):
    n_class = 2
    batch_size = input_data.shape[0]

    return np.random.random((batch_size, n_class))


def test_rise_function():
    # shape is batch, y, x, channel
    input_data = np.random.random((1, 224, 224, 3))

    heatmaps = dianna.explain(run_model, input_data, method="RISE", n_masks=200)

    assert heatmaps[0].shape == input_data[0].shape[:2]


def test_rise_filename():
    model_filename = 'tests/test_data/mnist_model.onnx'
    input_data = generate_data(batch_size=1)

    heatmaps = dianna.explain(model_filename, input_data, method="RISE", n_masks=200)

    assert heatmaps[0].shape == input_data[0].shape[:2]
