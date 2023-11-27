import numpy as np
import pandas
import dianna
from dianna.methods.rise_timeseries import RISETimeseries
from tests.methods.time_series_test_case import average_temperature_timeseries_with_1_cold_and_1_hot_day
from tests.methods.time_series_test_case import run_expert_model_3_step
from tests.utils import run_model


def test_rise_timeseries_correct_output_shape():
    """Test if rise runs and outputs the correct shape given some data and a model function."""
    input_data = np.random.random((10, 1))
    axis_labels = ['t', 'channels']
    labels = [1]

    heatmaps = dianna.explain_timeseries(run_model,
                                         input_data,
                                         'RISE',
                                         labels,
                                         axis_labels=axis_labels,
                                         n_masks=200,
                                         p_keep=.5)

    assert heatmaps.shape == (len(labels), *input_data.shape)


def test_rise_timeseries_with_expert_model_for_correct_max_and_min():
    """Test if RISE highlights the correct areas for this artificial example."""
    hot_day_index = 1
    cold_day_index = 2
    series_length = 4
    temperature_timeseries = average_temperature_timeseries_with_1_cold_and_1_hot_day(
        cold_day_index, hot_day_index, series_length=series_length)

    # summer_explanation, winter_explanation = dianna.explain_timeseries(
    #     # run_expert_model,
    #     run_continuous_expert_model,
    #     timeseries_data=temperature_timeseries,
    #     method='rise',
    #     labels=[0, 1],
    #     p_keep=0.1,
    #     n_masks=50,
    #     feature_res=series_length,
    #     mask_type=input_train_mean)

    explainer = RISETimeseries(n_masks=100000,
                               p_keep=0.5,
                               feature_res=series_length)
    summer_explanation, winter_explanation = explainer.explain(
        run_expert_model_3_step, temperature_timeseries, labels=[0, 1])
    print('\n')
    print(pandas.DataFrame(temperature_timeseries))
    dot = (explainer.masks[:, :, 0].T * explainer.predictions[:, 0]).T
    log = pandas.DataFrame(
        np.column_stack((explainer.masks[:, :, 0], explainer.masked[:, :, 0],
                         explainer.predictions, dot)),
        columns=[f'm{i}' for i in range(len(summer_explanation))] +
        [f'd{i}' for i in range(len(summer_explanation))] + ['P(S)', 'P(W)'] +
        [f'S{i}' for i in range(len(summer_explanation))])
    print(log)
    print(log.sum())

    print('\n')
    length = len(summer_explanation)
    margin = '      '
    print(
        margin, ' '.join(
            ['hot ' if i == hot_day_index else '    ' for i in range(length)]))
    print(
        margin, ' '.join([
            'cold' if i == cold_day_index else '    ' for i in range(length)
        ]))
    print(margin, ' '.join([f'{i: 4d}' for i in range(length)]))
    _print_series('summer', summer_explanation)
    _print_series('winter', winter_explanation)

    assert np.argmin(winter_explanation) == hot_day_index
    assert np.argmax(summer_explanation) == hot_day_index
    assert np.argmin(summer_explanation) == cold_day_index
    assert np.argmax(winter_explanation) == cold_day_index


def _print_series(title, series):
    mini = np.min(series)
    maxi = np.max(series)
    print(
        title, ' '.join([
            f'{x:0.1f}' + ('m' if x == mini else 'M' if x == maxi else ' ')
            for x in series[:, 0]
        ]))
