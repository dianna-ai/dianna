import numpy as np


class LIME:
    """
    Dummy class for testing.
    """
    def __init__(self, lime_arg_1=0, lime_arg_2=0):
        """
        Example constructor for testing.
        """
        self.lime_arg_1 = lime_arg_1
        self.lime_arg_2 = lime_arg_2

    def __call__(self, model, input_data):
        """
        Example call function.
        """
        self.model = model

        return np.zeros(input_data.shape, dtype=float)
