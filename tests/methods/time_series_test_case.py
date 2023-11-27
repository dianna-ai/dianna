"""Test case for timeseries xai methods.

This test case is designed to show if the xai methods could provide reasonable results.
In this test case, every test instance is a 28 days by 1 channel array indicating the max temp on a day.
"""
import numpy as np


MEAN_TEMP = 10


def input_train_mean(_data):
    """Return overall mean temperature of 14."""
    return MEAN_TEMP


def average_temperature_timeseries_with_1_cold_and_1_hot_day(
        cold_day_index, hot_day_index, series_length=28):
    """Creates a temperature time series of all 14s and a single cold (-10) and hot (30) day."""
    temperature_timeseries = np.expand_dims(np.zeros(series_length),
                                            axis=1) + MEAN_TEMP
    temperature_timeseries[hot_day_index] = 30
    temperature_timeseries[cold_day_index] = -10
    return temperature_timeseries


def run_expert_model(data):
    """A simple model that classifies a batch of timeseries.

    All instances with an average above MEAN_TEMP are classified as summer (0) and the rest as winter (1).
    """
    # Make actual decision
    is_summer = np.mean(np.mean(data, axis=1), axis=1) > MEAN_TEMP

    # Create the correct output format
    number_of_classes = 2
    number_of_instances = data.shape[0]
    result = np.zeros((number_of_instances, number_of_classes))
    result[is_summer] = [1.0, 0.0]
    result[~is_summer] = [0.0, 1.0]

    return result


def run_expert_model_3_step(data):
    """A simple model that classifies a batch of timeseries.

    All instances with an average above MEAN_TEMP are classified as summer (0) and the rest as winter (1).
    """
    # Make actual decision
    is_summer = np.mean(np.mean(data, axis=1), axis=1) > MEAN_TEMP
    is_winter = np.mean(np.mean(data, axis=1), axis=1) < MEAN_TEMP

    # Create the correct output format
    number_of_classes = 2
    number_of_instances = data.shape[0]
    result = np.zeros((number_of_instances, number_of_classes)) + 0.5
    result[is_summer] = [1.0, 0.0]
    result[is_winter] = [0.0, 1.0]

    return result


def run_continuous_expert_model(data):
    """A simple model that classifies a batch of timeseries.

    All instances with an average above 14 are classified as summer (0) and the rest as winter (1).
    """
    # Make actual decision
    np.mean(np.mean(data, axis=1), axis=1) > MEAN_TEMP
    diff = np.arctan(np.mean(np.mean(data, axis=1), axis=1) -
                     MEAN_TEMP) / np.pi

    # Create the correct output format
    number_of_classes = 2
    number_of_instances = data.shape[0]
    result = np.zeros((number_of_instances, number_of_classes))
    result[:, 0] = diff + 0.5
    result[:, 1] = -diff + 0.5

    # print('/n')
    # print(pandas.DataFrame(np.column_stack((data[:, :, 0], result))))

    return result
