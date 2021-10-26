import numpy as np
import torch
from captum.attr import DeepLift as DeepLift_captum


class DeepLift:
    """
    DeepLift implementation based on captum https://github.com/pytorch/captum
    """

    def __init__(self,
                 multiply_by_inputs=True
                 ):
        """DeepLift initializer.

        Args:
            multiply_by_inputs (bool, optional): Indicates whether to factor model inputs’ multiplier in the final attribution scores.
        """
        self.multiply_by_inputs = multiply_by_inputs

    def explain_image(self,
                      model,
                      input_data,
                      baselines=None,
                      label=1,
                      additional_forward_args=None,
                      return_convergence_delta=False,
                      custom_attribution_func=None
                      ):  # pylint: disable=too-many-arguments
        """
        Run the DeepLift explainer.

        Args:
            model (str): The path to a ONNX model on disk.
            input_data (np.ndarray): Data to be explained
            baselines (np.ndarray): Baselines define reference samples that are compared with the inputs.
            label (int): Index of class to be explained
            additional_forward_args (any, optional):
            return_convergence_delta (bool, optional):
            custom_attribution_func (callable, optional):

        Returns:
            list of (word, index of word in raw text, importance for target class) tuples
        """
        # call the deeplift class
        deeplift = DeepLift_captum(model)
        # convert input data from numpy array to torch.tensor
        input_data = torch.from_numpy(input_data).type(torch.FloatTensor)
        input_data.requires_grad = True
        if baselines is None:
            baselines = np.zeros_like(input_data)
        baselines = torch.from_numpy(baselines).type(torch.FloatTensor)
        # ensure the label is a native Python int
        label = int(label)
        # call the attributions
        attributions = deeplift.attribute(input_data,
                                          baselines,
                                          target=label,
                                          additional_forward_args=additional_forward_args,
                                          return_convergence_delta=return_convergence_delta,
                                          custom_attribution_func=custom_attribution_func
                                          )
        return attributions.cpu().detach().numpy()
