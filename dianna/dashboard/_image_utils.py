import numpy as np
from PIL import Image

# Per-channel statistics used by models trained on ImageNet (e.g. ResNet)
# See: https://github.com/onnx/models/tree/main/vision/classification/resnet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def input_layout(model):
    """Read (channels, height, width, channels_first) from the first input of an ONNX model.

    Height and width are 0 if the model accepts any size.
    """
    _, *dims = [
        d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim
    ]
    if len(dims) != 3:
        raise ValueError(
            f'Expected a model input of shape (batch, ...3 dims), got {len(dims) + 1} dims'
        )
    # ponytail: guesses the channels axis by its size, a model with e.g. 4 channels or a 3 pixel wide input breaks this
    if dims[0] in (1, 3):
        channels, height, width = dims
        return channels, height, width, True
    if dims[2] in (1, 3):
        height, width, channels = dims
        return channels, height, width, False
    raise ValueError(
        f'Cannot find a channels axis of size 1 or 3 in model input shape {dims}'
    )


def open_image(file, model):
    """Open an image and shape it to the model's input.

    Returns the model input (channels axis where the model expects it, values in [0, 1])
    and the image for display ((height, width) or (height, width, 3), values in [0, 1]).
    """
    channels, height, width, channels_first = input_layout(model)
    im = Image.open(file).convert('L' if channels == 1 else 'RGB')
    if height and width:
        im = im.resize((width, height))
    display = np.asarray(im, dtype=np.float32) / 255
    data = display[..., None] if channels == 1 else display
    return (np.moveaxis(data, -1, 0) if channels_first else data), display


def imagenet_normalize(data, channels_axis):
    """Normalize RGB data in [0, 1] with the ImageNet mean and standard deviation."""
    shape = [1] * data.ndim
    shape[channels_axis] = 3
    return ((data - IMAGENET_MEAN.reshape(shape)) /
            IMAGENET_STD.reshape(shape)).astype(np.float32)
