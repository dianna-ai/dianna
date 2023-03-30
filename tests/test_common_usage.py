import numpy as np
import dianna
import dianna.visualization
from tests.utils import run_model


def test_common_RISE_image_pipeline():  # noqa: N802 ignore case
    """No errors thrown while creating a relevance map and visualizing it."""
    input_image = np.random.random((224, 224, 3))
    axis_labels = {-1: 'channels'}
    labels = [0, 1]

    heatmap = dianna.explain_image(run_model, input_image, "RISE", labels, axis_labels=axis_labels)[0]
    dianna.visualization.plot_image(heatmap, show_plot=False)
    dianna.visualization.plot_image(heatmap, original_data=input_image[0], show_plot=False)


def test_common_RISE_timeseries_pipeline():  # noqa: N802 ignore case
    """No errors thrown while creating a relevance map and visualizing it."""
    input_timeseries = np.random.random((31, 1))
    labels = [0]

    heatmap = dianna.explain_timeseries(run_model, input_timeseries, "RISE", labels)[0]
    heatmap_channel = heatmap[:, 0]
    segments = []
    for i in range(len(heatmap_channel) - 1):
        segments.append({
            'index': i,
            'start': i,
            'stop': i + 1,
            'weight': heatmap_channel[i]})
    dianna.visualization.plot_timeseries(range(len(heatmap_channel)), input_timeseries[:, 0], segments, show_plot=False)
