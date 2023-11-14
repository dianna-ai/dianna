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

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to the
        means and stds in the training data.

        For categorical features, perturb by sampling according to the training
        distribution, and making a binary feature that is 1 when the value is the
        same as the instance being explained.

        More information can be found in the API guide:
        https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular

        Args:
            training_data (np.array): numpy 2d array
            mode (str, optional): "classification" or "regression"
            feature_names (strings, optional): list of names corresponding to the columns
                           in the training data.
            categorical_features (ints, optional): list of indices corresponding to the
                                                   categorical columns. Values in these
                                                   columns MUST be integers.
            kernel_width (int, optional): kernel width
            kernel (callable, optional): kernel
            verbose (bool, optional): verbose
            class_names (str, optional): list of class names, ordered according to whatever
                                         the classifier is using. If not present, class names
                                         will be '0', '1', ...
            feature_selection (str, optional): feature selection
            discretize_continuous (bool, optional): if True, all non-categorical features
                                                    will be discretized into quartiles.
            random_state (int or np.RandomState, optional): seed or random state
            kwargs: These parameters are passed on

        """
        init_instance_kwargs = utils.get_kwargs_applicable_to_function(
            LimeTabularExplainer, kwargs)

        self.explainer = LimeTabularExplainer(
            training_data,
            mode=mode,
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
            labels (Iterable(int), optional): Indices of classes to be explained.
            top_labels (str, optional): Top labels
            num_features (int, optional): Number of features
            num_samples (int, optional): Number of samples
            kwargs: These parameters are passed on

        Other keyword arguments: see the documentation for LimeTabularExplainer.explain_instance:
        https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_tabular.LimeTabularExplainer.explain_instance

        Returns:
            explanation: An Explanation object containing the LIME explanations for each class.
        """
        # run the explanation.
        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(
            self.explainer.explain_instance, kwargs)
        runner = utils.get_function(model_or_function)

        explanation = self.explainer.explain_instance(
            input_data,
            runner,
            labels=labels,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples,
            **explain_instance_kwargs,
        )

        return explanation
