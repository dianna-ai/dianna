import numpy as np

from dianna.utils.onnx_runner import SimpleModelRunner


def generate_data(batch_size):
    return np.random.randint(0, 256, size=(batch_size, 1, 28, 28))  # MNIST shape


def test_onnx_runner():
    filename = 'tests/test_data/mnist_model.onnx'
    n_classes = 2  # binary MNIST model
    batch_size = 3

    runner = SimpleModelRunner(filename)
    pred_onnx = runner(generate_data(batch_size))

    assert pred_onnx.shape == (batch_size, n_classes)
