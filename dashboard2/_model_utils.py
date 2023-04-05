import numpy as np
import onnx
from onnx_tf.backend import prepare


def preprocess_function(image):
    """For LIME: we divided the input data by 256 for the model (binary mnist) and LIME needs RGB values."""
    return (image / 256).astype(np.float32)


def fill_segmentation(values, segmentation):
    """For KernelSHAP: fill each pixel with SHAP values."""
    out = np.zeros(segmentation.shape)
    for i, _ in enumerate(values):
        out[segmentation == i] = values[i]
    return out


def load_model(file):
    onnx_model = onnx.load(file)
    return onnx_model


def load_labels(file):
    labels = [line.decode().rstrip() for line in file.readlines()]
    if labels is None or labels == ['']:
        raise ValueError(labels)
    return labels


def predict(*, model, image, labels):
    show_top = 2
    output_node = prepare(model, gen_tensor_dict=True).outputs[0]
    predictions = (prepare(model).run(image[None, ...])[output_node])
    preds = np.array(predictions[0])

    # get the predicted class
    labels[np.argmax(preds)]
    # get the top most likely results
    if show_top > len(labels):
        show_top = len(labels)
    # make sure the top results are ordered most to least likely
    ind = np.array(np.argpartition(preds, -show_top)[-show_top:])
    ind = ind[np.argsort(preds[ind])]
    ind = np.flip(ind)
    top = [labels[i] for i in ind]
    len(top)
