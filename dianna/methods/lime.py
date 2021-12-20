from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer
from dianna import utils


class LIME:
    """
    LIME implementation as wrapper around https://github.com/marcotcr/lime
    """

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
                 preprocess_function=None,
                 ):  # pylint: disable=too-many-arguments
        """LIME initializer.

        Args:
            kernel_width (int, optional):
            kernel (callable, optional):
            verbose (bool, optional):
            class_names (list, optional):
            feature_selection (str, optional):
            split_expression (regexp, optional):
            bow (bool, optional):
            mask_string (str, optional):
            random_state (int or np.RandomState, optional):
            char_level (bool, optional):
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
                      label=1,
                      top_labels=None,
                      num_features=10,
                      num_samples=5000,
                      positive_only=False,
                      hide_rest=True,
                      **kwargs,
                      ):  # pylint: disable=too-many-arguments
        """
        Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained
            label (int): Index of class to be explained
        Other keyword arguments: see the LIME documentation for LimeImageExplainer.explain_instance and
        ImageExplanation.get_image_and_mask:

        - https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer.explain_instance
        - https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.ImageExplanation.get_image_and_mask

        Returns:
            list of (word, index of word in raw text, importance for target class) tuples
        """
        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        # remove batch axis from input data; this is only here for a consistent API
        # but LIME wants data without batch axis
        if not len(input_data) == 1:
            raise ValueError("Length of batch axis must be one.")
        input_data = input_data[0]
        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(self.image_explainer.explain_instance, kwargs)
        explanation = self.image_explainer.explain_instance(input_data,
                                                            runner,
                                                            labels=(label,),
                                                            top_labels=top_labels,
                                                            num_features=num_features,
                                                            num_samples=num_samples,
                                                            **explain_instance_kwargs,
                                                            )

        get_image_and_mask_kwargs = utils.get_kwargs_applicable_to_function(explanation.get_image_and_mask, kwargs)
        mask = explanation.get_image_and_mask(label, positive_only=positive_only, hide_rest=hide_rest,
                                              num_features=num_features, **get_image_and_mask_kwargs)[1]
        return mask
