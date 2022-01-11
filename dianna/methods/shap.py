import numpy as np
import shap
import onnx
import skimage.segmentation
from onnx_tf.backend import prepare  # onnx to tf model converter&runner
import warnings
from dianna import utils


class KernelSHAP:
    """
    Kernel SHAP implementation based on shap https://github.com/slundberg/shap
    """

    def __init__(self):
        """KernelSHAP initializer.

        """
        
    @staticmethod
    def _segment_image(
        image,
        n_segments,
        compactness,
        sigma,
        channel_axis_first,
        **kwargs
    ):
        """Create segmentation to explain by segment, not every pixel

        This could help speed-up the calculation when the input size is very large.

        This function segments image using k-means clustering in Color-(x,y,z) space. It uses
        scikit-image.

        Args:
            Check keyword arguments for the skimage.segmentation.slic function
            via the following link:
            https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        """

        if channel_axis_first:
            image = np.transpose(image, (1, 2, 0))

        image_segments = skimage.segmentation.slic(
            image=image,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            **kwargs
        )

        return image_segments

    def explain_image(
        self,
        model,
        input_data,
        nsamples="auto",
        background=None,
        n_segments=100,
        compactness=10.0,
        sigma=0,
        channel_axis_first=False,
        **kwargs,
    ):
        """Run the KernelSHAP explainer.
           The model will be called with the function of image segmentation.

        Args:
            model (str): The path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained. It is mandatory to only
                                     provide a single example as input. This is because
                                     LIME / KernelShap is generally used for sample-based
                                     interpretability, training a separate interpretable
                                     model to explain a model’s prediction on each individual
                                     example. The input dimension must be
                                     [height, width, color_channels] or
                                     [color_channels, height, width] (see channel_axis_first)
            nsamples ("auto" or int): Number of times to re-evaluate the model when
                                      explaining each prediction. More samples lead
                                      to lower variance estimates of the SHAP values.
                                      The "auto" setting uses
                                      `nsamples = 2 * X.shape[1] + 2048`
            background (int): Background color for the masked image
            n_segments (int): The (approximate) number of labels in the segmented output image
            compactness (int): Balances color proximity and space proximity. Higher values give
                               more weight to space proximity, making superpixel shapes more
                               square/cubic.
            sigma (float): Width of Gaussian smoothing kernel for pre-processing for
                           each dimension of the image. Zero means no smoothing.
            channel_axis_first (boolean): If True, the shape of the input image is
                                          [color_channels, height, width]

        Other keyword arguments: see the documentation of kernel explainer of SHAP
                                 (also in function "shap_values") via:
        https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
                                 and the documentation of image segmentation via:
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic

        Returns:
            Explanation heatmap of shapley values for each class (np.ndarray).
        """
        self.onnx_model, self.input_node_dtype,\
        self.output_node = utils.onnx_model_node_loader(model)
        self.input_data = input_data
        self.channel_axis_first = channel_axis_first
        self.background = background

        # other keyword arguments for the method segment_image
        slic_kwargs = utils.get_kwargs_applicable_to_function(skimage.segmentation.slic, kwargs)

        # first check the dimension of input_data
        if input_data.ndim != 3:
            raise IOError(
                "The input image must follow the required shape [height, width, color_channels] or [color_channels, height, width]"
            )

        # call the segment method to create segmentation of input image
        self.image_segments = self._segment_image(
            input_data,
            n_segments,
            compactness,
            sigma,
            channel_axis_first,
            **slic_kwargs
        )

        # call the Kernel SHAP explainer
        explainer = shap.KernelExplainer(self._runner, np.zeros((1, n_segments)))

        with warnings.catch_warnings():
            # avoid warnings due to version conflicts
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(
                np.ones((1, n_segments)), nsamples=nsamples
            )

        return shap_values, self.image_segments

    def _mask_image(
        self, features, segmentation, image, background=None,
        channel_axis_first=False, datatype = np.float32
    ):
        """Define a function that depends on a binary mask representing
           if an image region is hidden.

        Args:
            features (np.ndarray): A matrix of samples (# samples x # features)
                                   on which to explain the model's output.
            segmentation (np.ndarray): Image segmentations generated by
                                       the function _segment_image
            image (np.ndarray): Image to be explained
            background (int): Background color for the masked image
            channel_axis (int): See the function explain_image
        """
        # if the image shape is [color_channels, height, width]
        if channel_axis_first:
            image = np.transpose(image, (1, 2, 0))
        # check the background color
        if background is None:
            background = image.mean((0, 1))

        # Create an empty 4D array
        out = np.zeros(
            (features.shape[0], image.shape[0], image.shape[1], image.shape[2])
        )

        for i in range(features.shape[0]):
            out[i, :, :, :] = image
            for j in range(features.shape[1]):
                if features[i, j] == 0:
                    out[i][segmentation == j, :] = background
        if channel_axis_first:
            out = np.transpose(out, (0, 3, 1, 2))

        return out.astype(datatype)

    def _runner(self, features):
        """Define a runner/wrapper to load models and values

        Args:
            features (np.ndarray): A matrix of samples (# samples x # features)
                                   on which to explain the model's output.
        """
        return prepare(self.onnx_model).run(
            self._mask_image(
                features,
                self.image_segments,
                self.input_data,
                self.background,
                self.channel_axis_first,
                self.input_node_dtype.as_numpy_dtype
            )
        )[f"{self.output_node}"]
