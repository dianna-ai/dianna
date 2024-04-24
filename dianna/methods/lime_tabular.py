"""LIME tabular explainer."""
import sys
from typing import Iterable
from typing import List
from typing import Union
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from dianna import utils


class LIMETabular:
    """Wrapper around the LIME explainer for tabular data."""

    def __init__(
        self,
        training_data: np.array,
        mode: str = "classification",
        feature_names: List[int] = None,
        categorical_features: List[int] = None,
        kernel_width: int = 25,
        kernel: callable = None,
        verbose: bool = False,
        class_names: List[str] = None,
        feature_selection: str = "auto",
        random_state: int = None,
        **kwargs,
    ) -> None:
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
            feature_names (list(str), optional): list of names corresponding to the columns
                           in the training data.
            categorical_features (list(int), optional): list of indices corresponding to the
                                                   categorical columns. Values in these
                                                   columns MUST be integers.
            kernel_width (int, optional): kernel width
            kernel (callable, optional): kernel
            verbose (bool, optional): verbose
            class_names (str, optional): list of class names, ordered according to whatever
                                         the classifier is using. If not present, class names
                                         will be '0', '1', ...
            feature_selection (str, optional): feature selection
            random_state (int or np.RandomState, optional): seed or random state
            kwargs: These parameters are passed on

        """
        self.mode = mode
        init_instance_kwargs = utils.get_kwargs_applicable_to_function(
            LimeTabularExplainer, kwargs)

        # temporary solution for setting num_features and top_labels
        self.num_features = len(feature_names)

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
        model_or_function: Union[str, callable],
        input_tabular: np.array,
        labels: Iterable[int],
        num_samples: int = 5000,
        **kwargs,
    ) -> np.array:
        """Run the LIME explainer.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained
                                                 or the path to a ONNX model on disk.
            input_tabular (np.ndarray): Data to be explained.
            labels (Iterable(int)): Indices of classes to be explained.
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
            input_tabular,
            runner,
            labels=labels,
            top_labels=sys.maxsize,
            num_features=self.num_features,
            num_samples=num_samples,
            **explain_instance_kwargs,
        )

        if self.mode == 'regression':
            local_exp = sorted(explanation.local_exp[1])
            saliency = [i[1] for i in local_exp]

        elif self.mode == 'classification':
            # extract scores from lime explainer
            saliency = []
            for i in range(len(explanation.local_exp.items())):
                local_exp = sorted(explanation.local_exp[i])
                # shape of local_exp [(index, saliency)]
                selected_saliency = [x[1] for x in local_exp]
                saliency.append(selected_saliency[:])

        else:
            raise ValueError(f'Unsupported mode "{self.mode}"')

        return np.array(saliency)
