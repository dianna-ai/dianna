import numpy as np


class SHAP:
    """
    Dummy class for testing.
    """
    def __init__(self, shap_arg_1=0, shap_arg_2=0):
        """
        Example constructor for testing.
        """
        self.shap_arg_1 = shap_arg_1
        self.shap_arg_2 = shap_arg_2

    def explain_image(self, model, input_data):
        """
        Example call function.
        """
        self.model = model

        return np.zeros(input_data.shape, dtype=float)
