import numpy as np

import dianna
from tests.utils import run_model


def test_rise_timeseries_correct_output_shape():
    """Test if rise runs and outputs the correct shape given some data and a model function."""
    input_data = np.random.random((10, 1))
    axis_labels = ['t', 'channels']
    labels = [1]

    heatmaps = dianna.explain_timeseries(run_model, input_data, "RISE", labels, axis_labels=axis_labels,
                                         n_masks=200, p_keep=.5)

    assert heatmaps.shape == (len(labels), *input_data.shape)
