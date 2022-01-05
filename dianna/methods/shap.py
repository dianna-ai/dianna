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

        """
        
    def _segment_image(self, image, n_segments, compactness, max_num_iter, sigma,
                       spacing, multichannel, convert2lab, enforce_connectivity,
                       min_size_factor, max_size_factor, slic_zero, start_label,
                       mask):
        """Create segmentation to explain by segment, not every pixel

        This could help speed-up the calculation when the input size is very large.

        This function segments image using k-means clustering in Color-(x,y,z) space. It uses
        scikit-image.

        Args:
            Check keyword arguments for the skimage.segmentation.slic function
            via the following link:
            https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        """
        
        image_segments = slic(image, n_segments, compactness, max_num_iter, sigma,
                              spacing, multichannel, convert2lab, enforce_connectivity,
                              min_size_factor, max_size_factor, slic_zero, start_label, 
                              mask)
        
        return image_segments

    def explain_image(self, model, input_data, nsamples='auto', background=None, # pylint: disable=too-many-arguments
                      n_segments=100, compactness=10.0, sigma=0, **kwargs):
        """Run the KernelSHAP explainer.
           The model will be called with the function of image segmentation.


        Args:
            model (str): The path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained. It is recommended to only
                                     provide a single example as input (tensors with
                                     first dimension or batch size = 1). This is because
                                     LIME / KernelShap is generally used for sample-based
                                     interpretability, training a separate interpretable
                                     model to explain a model’s prediction on each individual example.
            nsamples ("auto" or int): Number of times to re-evaluate the model when
                                      explaining each prediction. More samples lead
                                      to lower variance estimates of the SHAP values.
                                      The "auto" setting uses
                                      `nsamples = 2 * X.shape[1] + 2048`.

            background (int): background color for the masked image
            n_segments (int): the (approximate) number of labels in the segmented output image
            compactness (int): balances color proximity and space proximity. Higher values give
                               more weight to space proximity, making superpixel shapes more
                               square/cubic.
            sigma (float): Width of Gaussian smoothing kernel for pre-processing for
                           each dimension of the image. Zero means no smoothing.

        Other keyword arguments: see the documentation of kernel explainer of SHAP
                                 (also in function "shap_values") via:
        https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
                                 and the documentation of image segmentation via:
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
                                 
        Returns:
            Explanation heatmap of shapley values for each class (np.ndarray).
        """
        self.model = onnx.load(model)  # load onnx model
        self.input_data = input_data
        self.nsamples = nsamples
        self.background = background
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        # other keyword arguments for the method segment_image
        self.max_num_iter = kwargs.get("max_num_iter", 10)
        self.spacing = kwargs.get("spacing", None)
        self.multichannel = kwargs.get("multichannel", True)
        self.convert2lab = kwargs.get("convert2lab", None)
        self.enforce_connectivity = kwargs.get("enforce_connectivity", True)
        self.min_size_factor = kwargs.get("min_size_factor", 0.5)
        self.max_size_factor = kwargs.get("max_size_factor", 3)
        self.slic_zero = kwargs.get("slic_zero", False)
        self.start_label = kwargs.get("start_label", 1)
        self.mask = kwargs.get("mask", None)

        # call the segment method to create segmentation of input image
        self.image_segments = self._segment_image(self.input_data, self.n_segments,
                                                  self.compactness, self.max_num_iter,
                                                  self.sigma, self.spacing, self.multichannel,
                                                  self.convert2lab, self.enforce_connectivity,
                                                  self.min_size_factor, self.max_size_factor,
                                                  self.slic_zero, self.start_label, self.mask)

        #return np.zeros(input_data.shape, dtype=float)

    def _mask_image(self):
        """Define a function that depends on a binary mask representing
           if an image region is hidden
        """

    def _runner(self):
        """Define a runner/wrapper to load models and values

        """

