import numpy as np
import pytest
import dianna
import dianna.visualization
from tests.utils import get_dummy_model_function


def test_common_RISE_image_pipeline():  # noqa: N802 ignore case
    """No errors thrown while creating a relevance map and visualizing it."""
    input_image = np.random.random((5, 5, 3))
    axis_labels = {-1: 'channels'}
    labels = [0, 1]

    relevances = dianna.explain_image(get_dummy_model_function(n_outputs=2),
                                   input_image,
                                   'RISE',
                                   labels,
                                   axis_labels=axis_labels)[0]
    dianna.visualization.plot_image(relevances, show_plot=False)
    dianna.visualization.plot_image(relevances,
                                    original_data=input_image[0],
                                    show_plot=False)


@pytest.mark.parametrize('method, kwargs', [
    ('RISE', {'n_masks': 50, 'p_keep': .5}),
    ('LIME', {'num_samples': 20}),
    ('KernelSHAP', {'nsamples': 20, 'n_segments': 4}),
])
def test_greyscale_image_without_channels_axis(method, kwargs):
    """A 2D greyscale image is explained and the model receives it without channels axis (#410)."""
    input_image = np.random.random((8, 8)).astype(np.float32)

    def model(data):
        assert data.shape[1:] == input_image.shape
        return np.stack([data.mean(axis=(1, 2)), 1 - data.mean(axis=(1, 2))], axis=1)

    relevances = dianna.explain_image(model, input_image, method, [0], **kwargs)
    assert relevances.shape == (1, *input_image.shape)


def test_common_RISE_timeseries_pipeline():  # noqa: N802 ignore case
    """No errors thrown while creating a relevance map and visualizing it."""
    input_timeseries = np.random.random((31, 1))
    labels = [0]

    relevances = dianna.explain_timeseries(get_dummy_model_function(n_outputs=2),
                                        input_timeseries, 'RISE', labels)[0]
    segments = []
    for channel_number in range(relevances.shape[1]):
        relevances_channel = relevances[:, channel_number]
        for i in range(len(relevances_channel) - 1):
            segments.append({
                'index': i,
                'start': i,
                'stop': i + 1,
                'weight': relevances_channel[i],
                'channel': channel_number
            })
    r = range(len(relevances_channel))

    dianna.visualization.plot_timeseries(r,
                                         input_timeseries,
                                         segments,
                                         show_plot=False)
