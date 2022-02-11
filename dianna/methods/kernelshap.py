import warnings
import numpy as np
import shap
import skimage.segmentation
from dianna import utils


class KernelSHAP:
    """Kernel SHAP implementation based on shap https://github.com/slundberg/shap."""
    # axis labels required to be present in input image data
    required_labels = ('channels', )

    def __init__(self, axis_labels=None, preprocess_function=None):
        """Kernelshap initializer.

        Arguments:
            axis_labels (dict/list, optional): If a dict, key,value pairs of axis index, name.
                                               If a list, the name of each axis where the index
                                               in the list is the axis index
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.preprocess_function = preprocess_function
        self.axis_labels = axis_labels if axis_labels is not None else []
        # import here because it's slow
        from onnx_tf.backend import prepare  # pylint: disable=import-outside-toplevel
        self.onnx_to_tf = prepare

    @staticmethod
    def _segment_image(
        image,
        n_segments,
        compactness,
        sigma,
        **kwargs
    ):
        """Create segmentation to explain by segment, not every pixel.

        This could help speed-up the calculation when the input size is very large.

        This function segments image using k-means clustering in Color-(x,y,z) space. It uses
        scikit-image.

        Args:
            image (np.ndarray): Input image to be segmented.
            n_segments (int): The (approximate) number of labels in the segmented output image
            compactness (int): Balances color proximity and space proximity.
            sigma (float): Width of Gaussian smoothing kernel

            Check keyword arguments for the skimage.segmentation.slic function
            via the following link:
            https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        """
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
        labels=(0,),
        nsamples="auto",
        background=None,
        n_segments=100,
        compactness=10.0,
        sigma=0,
        **kwargs,
    ):  # pylint: disable=too-many-arguments
        """Run the KernelSHAP explainer.

        The model will be called with the function of image segmentation.

        Args:
            model (str): The path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained. It is mandatory to only
                                     provide a single example as input. This is because
                                     KernelShap is generally used for sample-based
                                     interpretability, training a separate interpretable
                                     model to explain a model prediction on each individual
                                     example. The input dimension must be
                                     [batch, height, width, color_channels] or
                                     [batch, color_channels, height, width] (see axis_labels)
            labels (tuple): Indices of classes to be explained
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
        self.labels = labels
        self.input_data = self._prepare_image_data(input_data)
        self.background = background

        # other keyword arguments for the method segment_image
        slic_kwargs = utils.get_kwargs_applicable_to_function(
            skimage.segmentation.slic, kwargs)

        # call the segment method to create segmentation of input image
        self.image_segments = self._segment_image(
            self.input_data,
            n_segments,
            compactness,
            sigma,
            **slic_kwargs
        )

        # call the Kernel SHAP explainer
        explainer = shap.KernelExplainer(
            self._runner, np.zeros((len(self.labels), n_segments)))

        with warnings.catch_warnings():
            # avoid warnings due to version conflicts
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(
                np.ones((len(self.labels), n_segments)), nsamples=nsamples
            )

        return shap_values, self.image_segments

    def _prepare_image_data(self, input_data):
        """Transforms the data to be of the shape and type KernelSHAP expects.

        Args:
            input_data (NumPy-compatible array): Data to be explained
        Returns:
            transformed input data
        """
        input_data = utils.to_xarray(
            input_data, self.axis_labels, KernelSHAP.required_labels)
        # ensure channels axis is last and keep track of where it was so we can move it back
        self.channels_axis_index = input_data.dims.index('channels')
        input_data = utils.move_axis(input_data, 'channels', -1)

        return input_data

    def _mask_image(
        self, features, segmentation, image, background=None,
        channels_axis_index=2, datatype=np.float32
    ):  # pylint: disable=too-many-arguments
        """Define a function that depends on a binary mask representing if an image region is hidden.

        Args:
            features (np.ndarray): A matrix of samples (# samples x # features)
                                   on which to explain the model's output.
            segmentation (np.ndarray): Image segmentations generated by
                                       the function _segment_image
            image (np.ndarray): Image to be explained
            background (int): Background color for the masked image
            channels_axis_index (int): See the function _prepare_image_data
            datatype (np.dtype): Datatype for the returned value
        """
        # check the background color
        if background is None:
            background = image.mean(axis=(0, 1))

        # Create an empty 4D array
        out = np.zeros(
            (features.shape[0], image.shape[0], image.shape[1], image.shape[2])
        )

        for i in range(features.shape[0]):
            out[i] = image
            for j in range(features.shape[1]):
                if features[i, j] == 0:
                    out[i][segmentation == j, :] = background

        # the output shape should satisfy the requirement from onnx model input shape
        if channels_axis_index != 2:
            out = np.transpose(out, (0, 3, 1, 2))

        return out.astype(datatype)

    def _runner(self, features):
        """Define a runner/wrapper to load models and values.

        Args:
            features (np.ndarray): A matrix of samples (# samples x # features)
                                   on which to explain the model's output.
        """
        model_input = self._mask_image(features,
                                       self.image_segments,
                                       self.input_data,
                                       self.background,
                                       self.channels_axis_index,
                                       self.input_node_dtype.as_numpy_dtype
                                       )
        if self.preprocess_function is not None:
            model_input = self.preprocess_function(model_input)
        return self.onnx_to_tf(self.onnx_model).run(model_input)[f"{self.output_node}"]
