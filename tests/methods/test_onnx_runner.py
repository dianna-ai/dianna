import numpy as np
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


def test_onnx_runner_multiple_inputs(tmp_path):
    """Tests if the onnx runner feeds all inputs of a model with multiple inputs."""
    filename = str(tmp_path / 'two_inputs.onnx')
    save_two_input_model(filename)
    image = np.ones((2, 28, 28, 1), dtype=np.float32)
    extra = np.array([[1, 2], [3, 4]], dtype=np.float32)

    pred = SimpleModelRunner(filename,
                             preprocess_function=lambda x: [x, extra])(image)

    assert np.array_equal(pred, 1 + extra)
