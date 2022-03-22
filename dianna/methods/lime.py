import numpy as np
from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer
from dianna import utils


class LIME:
    """Wrapper around the LIME explainer implemented by Marco Tulio Correia Ribeiro (https://github.com/marcotcr/lime)."""
    # axis labels required to be present in input image data
    required_labels = ('channels', )

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=False,
                 mask_string=None,
                 random_state=None,
                 char_level=False,
                 axis_labels=None,
                 preprocess_function=None,
                 ):  # pylint: disable=too-many-arguments
        """
        Initializes Lime explainer.

        Args:
            kernel_width (int, optional): kernel width
            kernel (callable, optional): kernel
            verbose (bool, optional): verbose
            class_names (list, optional): names of the classes
            feature_selection (str, optional): feature selection
            split_expression (regexp, optional): split expression
            bow (bool, optional): bow
            mask_string (str, optional): mask string
            random_state (int or np.RandomState, optional): seed or random state
            char_level (bool, optional): char level
            axis_labels (dict/list, optional): If a dict, key,value pairs of axis index, name.
                                               If a list, the name of each axis where the index
                                               in the list is the axis index
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.text_explainer = LimeTextExplainer(kernel_width,
                                                kernel,
                                                verbose,
                                                class_names,
                                                feature_selection,
                                                split_expression,
                                                bow,
                                                mask_string,
                                                random_state,
                                                char_level,
                                                )

        self.image_explainer = LimeImageExplainer(kernel_width,
                                                  kernel,
                                                  verbose,
                                                  feature_selection,
                                                  random_state,
                                                  )

        self.preprocess_function = preprocess_function
        self.axis_labels = axis_labels if axis_labels is not None else []

    def explain_text(self,
                     model_or_function,
                     input_data,
                     labels=(0,),
                     top_labels=None,
                     num_features=10,
                     num_samples=5000,
                     **kwargs,
                     ):  # pylint: disable=too-many-arguments
        """
        Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained
            labels ([int], optional): Iterable of indices of class to be explained

        Other keyword arguments: see the LIME documentation for LimeTextExplainer.explain_instance:
        https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_text.LimeTextExplainer.explain_instance.

        Returns:
            list of (word, index of word in raw text, importance for target class) tuples
        """
        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(self.text_explainer.explain_instance, kwargs)
        explanation = self.text_explainer.explain_instance(input_data,
                                                           runner,
                                                           labels=labels,
                                                           top_labels=top_labels,
                                                           num_features=num_features,
                                                           num_samples=num_samples,
                                                           **explain_instance_kwargs
                                                           )

        local_explanations = explanation.local_exp
        string_map = explanation.domain_mapper.indexed_string
        return [self._get_results_for_single_label(local_explanations[label], string_map) for label in labels]

    @staticmethod
    def _get_results_for_single_label(local_explanation, string_map):
        return [(string_map.word(index), int(string_map.string_position(index)), importance)
                for index, importance in local_explanation]

    def explain_image(self,
                      model_or_function,
                      input_data,
                      labels=(1,),
                      top_labels=None,
                      num_features=10,
                      num_samples=5000,
                      positive_only=False,
                      hide_rest=True,
                      **kwargs,
                      ):  # pylint: disable=too-many-arguments,too-many-locals
        """
        Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained. Must be an "RGB image", i.e. with values in
                                     the [0,255] range.
            labels (tuple): Indices of classes to be explained
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
        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(self.image_explainer.explain_instance, kwargs)
        explanation = self.image_explainer.explain_instance(input_data,
                                                            runner,
                                                            labels=labels,
                                                            top_labels=top_labels,
                                                            num_features=num_features,
                                                            num_samples=num_samples,
                                                            **explain_instance_kwargs,
                                                            )

        get_image_and_mask_kwargs = utils.get_kwargs_applicable_to_function(explanation.get_image_and_mask, kwargs)
        masks = [explanation.get_image_and_mask(label, positive_only=positive_only, hide_rest=hide_rest,
                                                num_features=num_features, **get_image_and_mask_kwargs)[1]
                 for label in labels]
        return masks

    def _prepare_image_data(self, input_data):
        """
        Transforms the data to be of the shape and type LIME expects.

        Args:
            input_data (NumPy-compatible array): Data to be explained

        Returns:
            transformed input data, preprocessing function to use with utils.get_function()
        """
        input_data = utils.to_xarray(input_data, self.axis_labels, LIME.required_labels)
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
        """
        Creates a full preprocessing function.

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
