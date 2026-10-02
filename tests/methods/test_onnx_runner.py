from unittest import mock
import numpy as np
import onnxruntime as ort
from dianna.utils.onnx_runner import SimpleModelRunner
from tests.utils import save_two_input_model


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


def test_onnx_runner_creates_session_once():
    """Tests that a runner creates its ONNX session once and reuses it."""
    filename = 'tests/test_data/mnist_model.onnx'
    batches = [generate_data(n).astype(np.float32) for n in (3, 1, 5)]

    with mock.patch.object(ort, 'InferenceSession', wraps=ort.InferenceSession) as m:
        runner = SimpleModelRunner(filename)
        for batch in batches:
            runner(batch)
    assert m.call_count == 1


def test_onnx_runner_multiple_inputs(tmp_path):
    """Tests if the onnx runner feeds all inputs of a model with multiple inputs."""
    filename = str(tmp_path / 'two_inputs.onnx')
    save_two_input_model(filename)
    image = np.ones((2, 28, 28, 1), dtype=np.float32)
    extra = np.array([[1, 2], [3, 4]], dtype=np.float32)

    pred = SimpleModelRunner(filename,
                             preprocess_function=lambda x: [x, extra])(image)

    assert np.array_equal(pred, 1 + extra)
