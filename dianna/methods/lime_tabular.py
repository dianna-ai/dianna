"""LIME tabular explainer."""

#import numpy as np
from lime.lime_tabular import LimeTabularExplainer


class LimeTabular:
    """Wrapper around the LIME explainer for tabular data."""

    def __init__(self,
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
            random_state (int or np.RandomState, optional): seed or random state

        """
        self.mode = mode
        self.explainer = LimeTabularExplainer(training_data,
                                              mode=self.mode,
                                              feature_names=feature_names,
                                              categorical_features=categorical_features,
                                              kernel_width=kernel_width,
                                              kernel=kernel,
                                              verbose=verbose,
                                              class_names=class_names,
                                              feature_selection=feature_selection,
                                              random_state=random_state,
                                              )

    def explain(self,
                model_or_function,
                input_data,
                labels=(1,),
                top_labels=None,
                num_features=10,
                num_samples=5000,
                **kwargs,
                ):
        # run the explanation.
        #explain_instance_kwargs = utils.get_kwargs_applicable_to_function(self.explainer.explain_instance, kwargs)
        #runner = utils.get_function(model_or_function, preprocess_function=full_preprocess_function)
        explanation = self.explainer.explain_instance(input_data,
                                                      model_or_function,
                                                      labels=labels,
                                                      top_labels=top_labels,
                                                      num_features=num_features,
                                                      num_samples=num_samples,
                                                      #**explain_instance_kwargs,
                                                      )
        
        return explanation
