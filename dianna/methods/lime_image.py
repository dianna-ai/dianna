"""LIME image explainer."""

import numpy as np
from lime.lime_image import ImageExplanation
from lime.lime_image import LimeImageExplainer
from dianna import utils


class LIMEImage:
    """Wrapper around the LIME explainer.

    Lime explainer byMarco Tulio Correia Ribeiro
    (https://github.com/marcotcr/lime).
    """

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 feature_selection='auto',
                 random_state=None,
                 axis_labels=None,
                 preprocess_function=None,
                 ):
        """Initializes Lime explainer.

        Args:
            kernel_width (int, optional): kernel width
            kernel (callable, optional): kernel
            verbose (bool, optional): verbose
            feature_selection (str, optional): feature selection
            random_state (int or np.RandomState, optional): seed or random state
            axis_labels (dict/list, optional): If a dict, key,value pairs of axis index, name.
                                               If a list, the name of each axis where the index
                                               in the list is the axis index
            preprocess_function (callable, optional): Function to preprocess input data with

        """
        self.preprocess_function = preprocess_function
        self.axis_labels = axis_labels if axis_labels is not None else []
        self.explainer = LimeImageExplainer(kernel_width,
                                            kernel,
                                            verbose,
                                            feature_selection,
                                            random_state,
                                            )

    def explain(self,
                model_or_function,
                input_data,
                labels,
                top_labels=None,
                num_features=10,
                num_samples=5000,
                return_masks=True,
                positive_only=False,
                hide_rest=True,
                **kwargs,
                ):
        """Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained. Must be an "RGB image", i.e. with values in
                                     the [0,255] range.
            labels (Iterable(int)): Indices of classes to be explained
            top_labels: Top labels
            num_features (int): Number of features
            num_samples (int): Number of samples
            return_masks (bool): If true, return discretized masks. Otherwise, return LIME scores
            positive_only (bool): Positive only
            hide_rest (bool): Hide rest
            kwargs: These parameters are passed on

        Other keyword arguments: see the LIME documentation for LimeImageExplainer.explain_instance and
        ImageExplanation.get_image_and_mask:

        - https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer.explain_instance
        - https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.ImageExplanation.get_image_and_mask

        Returns:
            list of heatmaps for each label
        """
        input_data, full_preprocess_function = self._prepare_image_data(input_data)
        runner = utils.get_function(model_or_function, preprocess_function=full_preprocess_function)

        # run the explanation.
        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(self.explainer.explain_instance, kwargs)
        explanation = self.explainer.explain_instance(input_data,
                                                      runner,
                                                      labels=labels,
                                                      top_labels=top_labels,
                                                      num_features=num_features,
                                                      num_samples=num_samples,
                                                      **explain_instance_kwargs,
                                                      )
        if return_masks:
            get_image_and_mask_kwargs = utils.get_kwargs_applicable_to_function(explanation.get_image_and_mask, kwargs)
            maps = [explanation.get_image_and_mask(label, positive_only=positive_only, hide_rest=hide_rest,
                                                   num_features=num_features, **get_image_and_mask_kwargs)[1]
                    for label in labels]
        else:
            maps = [self._get_explanation_values(label, explanation) for label in labels]
        return maps

    def _prepare_image_data(self, input_data):
        """Transforms the data to be of the shape and type LIME expects.

        Args:
            input_data (NumPy-compatible array): Data to be explained

        Returns:
            transformed input data, preprocessing function to use with utils.get_function()
        """
        # automatically determine the location of the channels axis if no axis_labels were provided
        axis_label_names = self.axis_labels.values() if isinstance(self.axis_labels, dict) else self.axis_labels
        if not axis_label_names:
            channels_axis_index = utils.locate_channels_axis(input_data.shape)
            self.axis_labels = {channels_axis_index: 'channels'}
        elif 'channels' not in axis_label_names:
            raise ValueError("When providing axis_labels it is required to provide the location"
                             " of the channels axis")

        input_data = utils.to_xarray(input_data, self.axis_labels)
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data.dims.index('channels')
        input_data = utils.move_axis(input_data, 'channels', -1)
        # LIME requires 3 channels. If the input has one channel, assume it is greyscale and
        # append two channel axes with identical data
        greyscale = False
        if len(input_data['channels']) == 1:
            greyscale = True
            input_data = input_data.sel(channels=0).expand_dims({'channels': 3}, axis=input_data.dims.index('channels'))
        # create preprocessing function that puts model input generated by LIME into the right shape and dtype,
        # followed by running the user's preprocessing function
        full_preprocess_function = self._get_full_preprocess_function(channels_axis_index, input_data.dtype,
                                                                      greyscale)
        # LIME requires float64 numpy data
        return input_data.values.astype(np.float64), full_preprocess_function

    def _get_full_preprocess_function(self, channel_axis_index, dtype, greyscale=False):
        """Creates a full preprocessing function.

        Creates a preprocessing function that incorporates both the (optional) user's
        preprocessing function, as well as any needed dtype and shape conversions

        Args:
            channel_axis_index (int): Axis index of the channels in the input data
            dtype (type): Data type of the input data (e.g. np.float32)
            greyscale (bool): Whether or not the data is greyscale (i.e. one channel)

        Returns:
            Function that first ensures the data has the same shape and type as the input data,
            then runs the users' preprocessing function
        """
        # LIME generates numpy arrays, so numpy is used to move the channel axis back to where it
        # was in the input data
        # one is added to the channels axis index because there is an extra first axis: the batch axis
        # if the data was greyscale, also remove the extra channels
        if greyscale:
            def moveaxis_function(data):
                return np.moveaxis(data[..., [0]], -1, channel_axis_index + 1).astype(dtype)
        else:
            def moveaxis_function(data):
                return np.moveaxis(data, -1, channel_axis_index + 1).astype(dtype)

        if self.preprocess_function is None:
            return moveaxis_function
        return lambda data: self.preprocess_function(moveaxis_function(data))

    def _get_explanation_values(self, label: int, explanation: ImageExplanation) -> np.array:
        """Get the importance scores from LIME in a salience map.

        Leverages the `ImageExplanation` class from LIME to generate salience maps.
        These salience maps are constructed using the segmentation masks from
        the explanation and fills these with the scores from the surrogate model
        (default for LIME is Ridge regression) used for the explanation.

        Args:
            label: The class label for the given explanation
            explanation: An Image Explanation generated by LIME

        Returns:
            A salience map containing the feature importances from LIME
        """
        class_explanation = explanation.local_exp[label]
        salience_map = np.zeros(explanation.segments.shape,
                                dtype=class_explanation[0][1].dtype) # Ensure same dataype for segment values

        # Fill segments
        for segment_id, segment_val in class_explanation:
            salience_map[segment_id == explanation.segments] = segment_val
        return salience_map
