from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer
from dianna.utils import get_function


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
                     distance_metric='cosine',
                     model_regressor=None,
                     ):  # pylint: disable=too-many-arguments
        """
        Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained
            labels ([int], optional): Iterable of indices of class to be explained
            top_labels ([type], optional):
            num_features (int, optional):
            num_samples (int, optional):
            distance_metric (str, optional):
            model_regressor ([type], optional):

        Returns:
            list of (word, index of word in raw text, importance for target class) tuples
        """
        runner = get_function(model_or_function, preprocess_function=self.preprocess_function)
        explanation = self.text_explainer.explain_instance(input_data,
                                                           runner,
                                                           labels,
                                                           top_labels,
                                                           num_features,
                                                           num_samples,
                                                           distance_metric,
                                                           model_regressor,
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
                      hide_color=None,
                      top_labels=None,
                      num_features=10,
                      num_samples=5000,
                      batch_size=10,
                      segmentation_fn=None,
                      distance_metric='cosine',
                      model_regressor=None,
                      random_seed=None,
                      ):  # pylint: disable=too-many-arguments,too-many-locals
        """
        Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained
            label (int): Index of class to be explained
            hide_color (float, optional):
            top_labels (int, optional):
            num_features (int, optional):
            num_samples (int, optional):
            batch_size (int, optional):
            segmentation_fn (callable, optional):
            distance_metric (str, optional):
            model_regressor ([type], optional):
            random_seed (int, optional):

        Returns:
            list of (word, index of word in raw text, importance for target class) tuples
        """
        runner = get_function(model_or_function, preprocess_function=self.preprocess_function)
        # remove batch axis from input data; this is only here for a consistent API
        # but LIME wants data without batch axis
        if not len(input_data) == 1:
            raise ValueError("Length of batch axis must be one.")
        input_data = input_data[0]
        explanation = self.image_explainer.explain_instance(input_data,
                                                            runner,
                                                            labels=(label,),
                                                            hide_color=hide_color,
                                                            top_labels=top_labels,
                                                            num_features=num_features,
                                                            num_samples=num_samples,
                                                            batch_size=batch_size,
                                                            segmentation_fn=segmentation_fn,
                                                            distance_metric=distance_metric,
                                                            model_regressor=model_regressor,
                                                            random_seed=random_seed
                                                            )

        mask = explanation.get_image_and_mask(label, positive_only=False, hide_rest=True, num_features=num_features)[1]
        return mask
