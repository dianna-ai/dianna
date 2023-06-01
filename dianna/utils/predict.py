import numpy as np
from tqdm import tqdm


def make_predictions(data, runner, batch_size):
    """Make predictions with the input data.

    Process the data with the model runner and return the predictions.

    Args:
        data (np.ndarray): An array of masked input data to be processed by the model.
        runner (object): An object that runs the model on the input data and returns predictions.
        batch_size (int): The number of masked inputs to process in each batch.

    Returns:
        np.ndarray: An array of predictions made by the model on the input data.
    """
    number_of_masks = len(data)
    predictions = []
    for i in tqdm(range(0, number_of_masks, batch_size), desc="Explaining"):
        predictions.append(runner(data[i : i + batch_size]))
    return np.concatenate(predictions)
