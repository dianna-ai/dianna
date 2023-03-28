import numpy as np
import dianna
from tests.methods.time_series_test_case import average_temperature_timeseries_with_1_cold_and_1_hot_day
from tests.methods.time_series_test_case import input_train_mean
from tests.methods.time_series_test_case import run_expert_model
from tests.utils import run_model


def test_rise_timeseries_correct_output_shape():
    """Test if rise runs and outputs the correct shape given some data and a model function."""
    input_data = np.random.random((10, 1))
    axis_labels = ['t', 'channels']
    labels = [1]

    heatmaps = dianna.explain_timeseries(run_model, input_data, "RISE", labels, axis_labels=axis_labels,
                                         n_masks=200, p_keep=.5)

    assert heatmaps.shape == (len(labels), *input_data.shape)


def test_rise_timeseries_with_expert_model_for_correct_max_and_min():
    """Test if RISE highlights the correct areas for this artificial example."""
    hot_day_index = 6
    cold_day_index = 12
    temperature_timeseries = average_temperature_timeseries_with_1_cold_and_1_hot_day(cold_day_index, hot_day_index)

    summer_explanation, winter_explanation = dianna.explain_timeseries(run_expert_model,
                                                                       timeseries_data=temperature_timeseries,
                                                                       method='rise',
                                                                       labels=[0, 1],
                                                                       p_keep=0.1, n_masks=10000,
                                                                       mask_type=input_train_mean)

    assert np.argmax(summer_explanation) == hot_day_index
    assert np.argmin(summer_explanation) == cold_day_index
    assert np.argmax(winter_explanation) == cold_day_index
    assert np.argmin(winter_explanation) == hot_day_index
