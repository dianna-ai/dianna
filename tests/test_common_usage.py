import numpy as np
import dianna
import dianna.visualization
from tests.utils import run_model


def test_common_RISE_image_pipeline():  # noqa: N802 ignore case
    """No errors thrown while creating a relevance map and visualizing it."""
    input_image = np.random.random((5, 5, 3))
    axis_labels = {-1: 'channels'}
    labels = [0, 1]

    heatmap = dianna.explain_image(run_model,
                                   input_image,
                                   'RISE',
                                   labels,
                                   axis_labels=axis_labels)[0]
    dianna.visualization.plot_image(heatmap, show_plot=False)
    dianna.visualization.plot_image(heatmap,
                                    original_data=input_image[0],
                                    show_plot=False)


def test_common_RISE_timeseries_pipeline():  # noqa: N802 ignore case
    """No errors thrown while creating a relevance map and visualizing it."""
    input_timeseries = np.random.random((31, 1))
    labels = [0]

    heatmap = dianna.explain_timeseries(run_model, input_timeseries, 'RISE',
                                        labels)[0]
    segments = []
    for channel_number in range(heatmap.shape[1]):
        heatmap_channel = heatmap[:, channel_number]
        for i in range(len(heatmap_channel) - 1):
            segments.append({
                'index': i,
                'start': i,
                'stop': i + 1,
                'weight': heatmap_channel[i],
                'channel': channel_number
            })
    r = range(len(heatmap_channel))

    dianna.visualization.plot_timeseries(r,
                                         input_timeseries,
                                         segments,
                                         show_plot=False)
