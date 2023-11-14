"""LIME tabular explainer."""

from lime.lime_tabular import LimeTabularExplainer
from dianna import utils


class LimeTabular:
    """Wrapper around the LIME explainer for tabular data."""

    def __init__(
        self,
        training_data,
        mode='classification',
        feature_names=None,
        categorical_features=None,
        kernel_width=25,
        kernel=None,
        verbose=False,
        class_names=None,
        feature_selection='auto',
        random_state=None,
        **kwargs,
    ):
        """Initializes Lime explainer.

        Args:
            kernel_width (int, optional): kernel width
            kernel (callable, optional): kernel
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            verbose (bool, optional): verbose
            class_names: list of class names, ordered according to whatever the classifier
                          is using. If not present, class names will be '0', '1', ...
            feature_selection (str, optional): feature selection
            discretize_continuous (bool, optional): if True, all non-categorical features
                                                    will be discretized into quartiles.
            random_state (int or np.RandomState, optional): seed or random state

        """
        init_instance_kwargs = utils.get_kwargs_applicable_to_function(
            LimeTabularExplainer, kwargs)

        self.mode = mode
        self.explainer = LimeTabularExplainer(
            training_data,
            mode=self.mode,
            feature_names=feature_names,
            categorical_features=categorical_features,
            kernel_width=kernel_width,
            kernel=kernel,
            verbose=verbose,
            class_names=class_names,
            feature_selection=feature_selection,
            random_state=random_state,
            **init_instance_kwargs,
        )

    def explain(
        self,
        model_or_function,
        input_data,
        labels=(1, ),
        top_labels=None,
        num_features=10,
        num_samples=5000,
        **kwargs,
    ):
        """Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained
                                                 or the path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained.
            labels (Iterable(int)): Indices of classes to be explained.
            top_labels: Top labels
            num_features (int): Number of features
            num_samples (int): Number of samples
            return_masks (bool): If true, return discretized masks. Otherwise, return LIME scores
            positive_only (bool): Positive only
            hide_rest (bool): Hide rest
            kwargs: These parameters are passed on

        Other keyword arguments: see the LIME documentation for LimeImageExplainer.explain_instance:

        https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer.explain_instance

        Returns:
            list of heatmaps for each label
        """
        # run the explanation.
        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(
            self.explainer.explain_instance, kwargs)
        #runner = utils.get_function(model_or_function, preprocess_function=full_preprocess_function)
        explanation = self.explainer.explain_instance(
            input_data,
            model_or_function,
            labels=labels,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples,
            **explain_instance_kwargs,
        )

        return explanation
