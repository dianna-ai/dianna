from typing import List
from typing import Optional
from typing import Union
import numpy as np
import shap
from shap import KernelExplainer
from dianna import utils


class KERNELSHAPTabular:
    """Wrapper around the SHAP Kernel explainer for tabular data."""

    def __init__(
        self,
        training_data: np.array,
        mode: str = "classification",
        feature_names: List[int] = None,
        training_data_kmeans: Optional[int] = None,
        silent: bool = False,
    ) -> None:
        """Initializer of KERNELSHAPTabular.

        Training data must be provided for the explainer to estimate the expected
        values.

        More information can be found in the API guide:
        https://github.com/shap/shap/blob/master/shap/explainers/_kernel.py

        Arguments:
            training_data (np.array): training data, which should be numpy 2d array
            mode (str, optional): "classification" or "regression"
            feature_names (list(str), optional): list of names corresponding to the columns
                                                 in the training data.
            training_data_kmeans(int, optional): summarize the whole training set with
                                                 weighted kmeans
            silent (bool, optional): whether to print progress messages
        """
        if training_data_kmeans:
            self.training_data = shap.kmeans(training_data,
                                             training_data_kmeans)
        else:
            self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.explainer: KernelExplainer
        self.silent = silent

    def explain(
        self,
        model_or_function: Union[str, callable],
        input_tabular: np.array,
        link: str = "identity",
        **kwargs,
    ) -> np.array:
        """Run the KernelSHAP explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained
                                                 or the path to a ONNX model on disk.
            input_tabular (np.ndarray): Data to be explained.
            link (str): A generalized linear model link to connect the feature importance values
                        to the model. Must be either "identity" or "logit".
            kwargs: These parameters are passed on

        Other keyword arguments: see the documentation for KernelExplainer:
        https://github.com/shap/shap/blob/master/shap/explainers/_kernel.py

        Returns:
            An array (np.ndarray) containing the KernelExplainer explanations for each class.
        """
        init_instance_kwargs = utils.get_kwargs_applicable_to_function(
            KernelExplainer, kwargs)
        self.explainer = KernelExplainer(model_or_function, self.training_data,
                                         link, **init_instance_kwargs)

        explain_instance_kwargs = utils.get_kwargs_applicable_to_function(
            self.explainer.shap_values, kwargs)

        saliency = self.explainer.shap_values(input_tabular, silent=self.silent, **explain_instance_kwargs)

        if self.mode == 'regression':
            saliency = saliency[0]

        return np.array(saliency)
