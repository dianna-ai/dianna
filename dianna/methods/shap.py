import numpy as np
import shap
import onnx
from skimage.segmentation import slic
from onnx_tf.backend import prepare # onnx to tf model converter&runner


class KernelSHAP:
    """
    Kernel SHAP implementation based on shap https://github.com/slundberg/shap
    """
    def __init__(self):
        """KernelSHAP initializer.

        Args:
            model (str): The path to a ONNX model on disk.
        """
        
    def segment_image(self, input_data):
        """Create segmentation to explain by segment, not every pixel

        Args:
            model (str): The path to a ONNX model on disk.
        """        
        

    def explain_image(self, model):
        """Run the RISE explainer.
           The model will be called with masked images,
           with a shape defined by `batch_size` and the shape of `input_data`

        Args:
            model (str): The path to a ONNX model on disk.

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        self.model = onnx.load(model)  # load onnx model

        return np.zeros(input_data.shape, dtype=float)

    def _mask_image(self):
        """Define a function that depends on a binary mask representing if an image region is hidden

        """

    def _runner(self):
        """Define a runner/wrapper to load models and values

        """

