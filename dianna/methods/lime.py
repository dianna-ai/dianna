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
        """
        self.explainer = LimeTextExplainer(kernel_width,
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

    def explain_text(self,
                     model_or_function,
                     input_data,
                     label=1,
                     top_labels=None,
                     num_features=10,
                     num_samples=5000,
                     distance_metric='cosine',
                     model_regressor=None,
                     ):  # pylint: disable=too-many-arguments
        """Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained
            label (int): Index of class to be explained
            top_labels ([type], optional):
            num_features (int, optional):
            num_samples (int, optional):
            distance_metric (str, optional):
            model_regressor ([type], optional):

        Returns:
            list of (word, index of word in raw text, importance for target class) tuples
        """
        runner = get_function(model_or_function)
        explanation = self.explainer.explain_instance(input_data,
                                                      runner,
                                                      (label,),
                                                      top_labels,
                                                      num_features,
                                                      num_samples,
                                                      distance_metric,
                                                      model_regressor,
                                                      )

        local_explanation = explanation.local_exp[label]
        string_map = explanation.domain_mapper.indexed_string
        return [(string_map.word(index), int(string_map.string_position(index)), importance)
                for index, importance in local_explanation]
