"""LIME text explainer."""
from lime.lime_text import LimeTextExplainer
from dianna import utils


class LIMEText:
    """Wrapper around the LIME explainer.

    See Lime explainer by Marco Tulio Correia Ribeiro
    (https://github.com/marcotcr/lime).
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
                 ):
        """Initializes Lime explainer.

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
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.preprocess_function = preprocess_function
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

    def explain(self,
                model_or_function,
                input_text,
                labels,
                tokenizer=None,
                top_labels=None,
                num_features=10,
                num_samples=5000,
                **kwargs,
                ):
        """Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            tokenizer : Tokenizer class with tokenize and convert_tokens_to_string methods, and mask_token attribute
            input_text (np.ndarray): Data to be explained
            labels (Iterable(int)): Iterable of indices of class to be explained
            top_labels: Top labels
            num_features (int): Number of features
            num_samples (int): Number of samples
            kwargs: These parameters are passed on

        Other keyword arguments: see the LIME documentation for LimeTextExplainer.explain_instance:
        https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_text.LimeTextExplainer.explain_instance.

        Returns:
            List of tuples (word, index of word in raw text, importance for target class) for each class
        """
        if tokenizer is None:
            raise ValueError('Please provide a tokenizer to explain_text.')

        self.explainer.split_expression = tokenizer.tokenize  # lime accepts a callable as a split_expression

        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(
            self.explainer.explain_instance, kwargs)
        explanation = self.explainer.explain_instance(input_text,
                                                      runner,
                                                      labels=labels,
                                                      top_labels=top_labels,
                                                      num_features=num_features,
                                                      num_samples=num_samples,
                                                      **explain_instance_kwargs
                                                      )

        local_explanations = explanation.local_exp
        string_map = explanation.domain_mapper.indexed_string
        return [self._reshape_result_for_single_label(
            local_explanations[label], string_map) for label in labels]

    @staticmethod
    def _reshape_result_for_single_label(local_explanation, string_map):
        """Get results for single label.

        Args:
            local_explanation: Lime output, map of tuples (index, importance)
            string_map: Lime's IndexedString, see documentation:
                https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=indexedstring#lime.lime_text.IndexedString
        """
        return [(string_map.word(index), index, importance) for index, importance in local_explanation]
