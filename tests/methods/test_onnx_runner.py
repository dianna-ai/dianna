import numpy as np
from dianna.utils.onnx_runner import SimpleModelRunner


def generate_data(batch_size):
    """Generate a batch of random data."""
    return np.random.randint(0, 256,
                             size=(batch_size, 1, 28, 28))  # MNIST shape


def test_onnx_runner():
    """Tests if the onnx runner can run a test model and make predictions."""
    filename = 'tests/test_data/mnist_model.onnx'
    n_classes = 2  # binary MNIST model
    batch_size = 3

    runner = SimpleModelRunner(filename)
    pred_onnx = runner(generate_data(batch_size).astype(np.float32))

    assert pred_onnx.shape == (batch_size, n_classes)


def test_onnx_runner_reused_session_matches_fresh():
    """Tests that repeated calls on one runner (cached session) match fresh runners."""
    filename = 'tests/test_data/mnist_model.onnx'
    batches = [generate_data(n).astype(np.float32) for n in (3, 1, 5)]

    runner = SimpleModelRunner(filename)
    for batch in batches:
        np.testing.assert_array_equal(runner(batch), SimpleModelRunner(filename)(batch))
