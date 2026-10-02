import numpy as np
from dianna import utils
from dianna.utils.maskers import generate_interpolated_float_masks_for_image
from dianna.utils.predict import make_predictions
from dianna.utils.rise_utils import saliency


class RISEImage:
    """RISE implementation for images based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb."""

    def __init__(
        self,
        n_masks=1000,
        feature_res=8,
        p_keep=None,
        axis_labels=None,
        preprocess_function=None,
    ):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of input data to keep in each mask (Default: auto-tune this value).
            axis_labels (dict/list, optional): If a dict, key,value pairs of axis index, name.
                                               If a list, the name of each axis where the index
                                               in the list is the axis index
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None
        self.fill = 0
        self.axis_labels = axis_labels if axis_labels is not None else []

    def explain(self,
                model_or_function,
                input_data,
                labels,
                batch_size=100,
                mask_type='mean'):
        """Runs the RISE explainer on images.

           The model will be called with masked images,
           with a shape defined by `batch_size` and the shape of `input_data`.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Image to be explained
            batch_size (int): Batch size to use for running the model.
            labels (Iterable(int)): Labels to be explained
            mask_type (str or callable): What masked pixels are replaced with. 'mean' (default: mean
                                         color per channel), 'black' (zeros), 'white' (the image's
                                         maximum value), 'blur' (Gaussian blurred image), 'noise' (the
                                         image's own pixels shuffled, so fill colors follow its color
                                         distribution) or a callable(image) returning a fill array
                                         broadcastable to the channels-last image.

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        input_data, runner = self._prepare_input_data_and_model(
            input_data, model_or_function)
        self.fill = self._get_fill(input_data.values[0], mask_type)

        active_p_keep = (self._determine_p_keep(input_data, runner)
                         if self.p_keep is None else self.p_keep)

        # data shape without batch axis and channel axis
        img_shape = input_data.shape[1:3]
        # Expose masks for to make user inspection possible
        self.masks = generate_interpolated_float_masks_for_image(
            img_shape, active_p_keep, self.n_masks, self.feature_res)

        masked = self._mask(input_data, self.masks)

        self.predictions = make_predictions(masked, runner, batch_size)

        # Reshape to (n_classes, *img_shape)
        result = saliency(self.predictions, self.masks,
                          active_p_keep).reshape(-1, *img_shape)
        if labels is not None:
            result = result[list(labels)]
        return result

    def _mask(self, input_data, masks):
        """Blend input with the fill value: fill where mask is 0, input where mask is 1."""
        return self.fill + (input_data - self.fill) * masks

    def _get_fill(self, image, mask_type):
        """Replacement values for masked pixels, for a channels-last image."""
        if callable(mask_type):
            return mask_type(image)
        if mask_type == 'black':
            return 0
        if mask_type == 'white':
            # ponytail: brightest value in the image, so it works for both 0-1 and 0-255 ranges
            return image.max()
        if mask_type == 'mean':
            return image.mean(axis=(0, 1))
        if mask_type == 'blur':
            from skimage.filters import gaussian
            # ponytail: sigma fixed at one mask cell; pass a callable for other blur strengths
            sigma = max(image.shape[:2]) / self.feature_res
            return gaussian(image,
                            sigma=sigma,
                            channel_axis=-1,
                            preserve_range=True)
        if mask_type == 'noise':
            # ponytail: one noise sample shared by all masks; per-mask noise if heatmaps show its pattern
            return np.random.permutation(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        raise ValueError(f'Unknown mask_type selected: {mask_type}')

    def _prepare_input_data_and_model(self, input_data, model_or_function):
        """Prepares the input data as an xarray with an added batch dimension and creates a preprocessing function."""
        self._set_axis_labels(input_data)
        input_data = utils.to_xarray(input_data, self.axis_labels)
        input_data = input_data.expand_dims('batch', 0)
        input_data, full_preprocess_function = self._prepare_image_data(
            input_data)
        runner = utils.get_function(
            model_or_function, preprocess_function=full_preprocess_function)
        return input_data, runner

    def _set_axis_labels(self, input_data):
        # automatically determine the location of the channels axis if no axis_labels were provided
        axis_label_names = (self.axis_labels.values() if isinstance(
            self.axis_labels, dict) else self.axis_labels)
        if not axis_label_names:
            channels_axis_index = utils.locate_channels_axis(input_data.shape)
            self.axis_labels = {channels_axis_index: 'channels'}
        elif 'channels' not in axis_label_names:
            raise ValueError(
                'When providing axis_labels it is required to provide the location'
                ' of the channels axis')

    def _determine_p_keep(self, input_data, runner, n_masks=100):
        """See n_mask default value https://github.com/dianna-ai/dianna/issues/24#issuecomment-1000152233."""
        p_keeps = np.arange(0.1, 1.0, 0.1)
        stds = []
        for p_keep in p_keeps:
            std = self._calculate_max_class_std(p_keep,
                                                runner,
                                                input_data,
                                                n_masks=n_masks)
            stds += [std]
        best_i = np.argmax(stds)
        best_p_keep = p_keeps[best_i]
        print(
            f'Rise parameter p_keep was automatically determined at {best_p_keep}'
        )
        return best_p_keep

    def _calculate_max_class_std(self, p_keep, runner, input_data, n_masks):
        img_shape = input_data.shape[1:3]
        masks = generate_interpolated_float_masks_for_image(
            img_shape, p_keep, n_masks, self.feature_res)
        masked = self._mask(input_data, masks)
        predictions = make_predictions(masked, runner, batch_size=50)
        std_per_class = predictions.std(axis=0)
        return np.max(std_per_class)

    def _prepare_image_data(self, input_data):
        """Transforms the data to be of the shape and type RISE expects.

        Args:
            input_data (xarray): Data to be explained

        Returns:
            transformed input data, preprocessing function to use with utils.get_function()
        """
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data.dims.index('channels')
        input_data = utils.move_axis(input_data, 'channels', -1)
        # create preprocessing function that puts model input generated by RISE into the right shape and dtype,
        # followed by running the user's preprocessing function
        full_preprocess_function = self._get_full_preprocess_function(
            channels_axis_index, input_data.dtype)
        return input_data, full_preprocess_function

    def _get_full_preprocess_function(self, channel_axis_index, dtype):
        """Creates a full preprocessing function.

        Creates a preprocessing function that incorporates both the (optional) user's
        preprocessing function, as well as any needed dtype and shape conversions

        Args:
            channel_axis_index (int): Axis index of the channels in the input data
            dtype (type): Data type of the input data (e.g. np.float32)

        Returns:
            Function that first ensures the data has the same shape and type as the input data,
            then runs the users' preprocessing function
        """

        def moveaxis_function(data):
            return (utils.move_axis(data, 'channels',
                                    channel_axis_index).astype(dtype).values)

        if self.preprocess_function is None:
            return moveaxis_function
        return lambda data: self.preprocess_function(moveaxis_function(data))
