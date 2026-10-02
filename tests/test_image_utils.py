"""Tests for the image loading of the dashboard, which does not need a browser."""
import sys
from pathlib import Path
import numpy as np
import pytest
from onnx import TensorProto
from onnx import helper
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / 'dianna' / 'dashboard'))
from _image_utils import imagenet_normalize  # noqa: E402
from _image_utils import open_image  # noqa: E402


def _model(shape):
    """Minimal ONNX model with an input of the given shape."""
    io = helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)
    return helper.make_model(
        helper.make_graph(
            [helper.make_node('Identity', ['x'], ['y'])], 'g', [io],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)]),
        ir_version=8,
        opset_imports=[helper.make_opsetid('', 13)])


@pytest.fixture
def image_file(tmp_path):
    """Random 50x40 RGB image on disk."""
    path = tmp_path / 'image.png'
    Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (50, 40, 3),
                                          dtype=np.uint8)).save(path)
    return path


@pytest.mark.parametrize('shape, expected', [
    (['N', 28, 28, 1], (28, 28, 1)),
    (['N', 3, 128, 128], (3, 128, 128)),
    (['N', 1, 64, 32], (1, 64, 32)),
    (['N', 3, 'h', 'w'], (3, 50, 40)),
])
def test_open_image_matches_model_input(image_file, shape, expected):
    """Image is resized and laid out as the model's input declares, in [0, 1]."""
    data, display = open_image(image_file, _model(shape))
    assert data.shape == expected
    assert data.dtype == np.float32
    assert 0 <= data.min() and data.max() <= 1
    assert display.shape[:2] == (data.shape[1:]
                                 if shape[1] in (1, 3) else data.shape[:2])


def test_open_image_unknown_layout(image_file):
    """A model input without a channels axis of size 1 or 3 is rejected."""
    with pytest.raises(ValueError):
        open_image(image_file, _model(['N', 4, 8, 8]))


def test_imagenet_normalize():
    """Normalisation is applied along the given channels axis."""
    batch = np.full((2, 5, 5, 3), 0.5, dtype=np.float32)
    channels_last = imagenet_normalize(batch, 3)
    channels_first = imagenet_normalize(np.moveaxis(batch, 3, 1), 1)
    np.testing.assert_allclose(channels_last[0, 0, 0],
                               (0.5 - np.array([0.485, 0.456, 0.406])) /
                               [0.229, 0.224, 0.225],
                               rtol=1e-6)
    np.testing.assert_allclose(np.moveaxis(channels_first, 1, 3),
                               channels_last)


def test_add_softmax():
    """The model with softmax added outputs probabilities of the original logits."""
    import onnxruntime as rt
    from _model_utils import add_softmax
    x = np.array([[1., -2., 3.], [0., 0., 0.]], dtype=np.float32)
    model = add_softmax(_model(['N', 3]))
    out = rt.InferenceSession(model.SerializeToString()).run(None, {'x': x})[0]
    np.testing.assert_allclose(out, np.exp(x) / np.exp(x).sum(1, keepdims=True), rtol=1e-6)
