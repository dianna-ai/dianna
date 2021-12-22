import numpy as np
import dianna
import dianna.visualization
from tests.utils import run_model


input_data = np.random.random((1, 224, 224, 3))
axes_labels = {0: 'batch', -1: 'channels'}


def test_common_RISE_pipeline():  # noqa: N802 ignore case
    heatmap = dianna.explain_image(run_model, input_data, method="RISE", axes_labels=axes_labels)[0]
    dianna.visualization.plot_image(heatmap, show_plot=False)
    dianna.visualization.plot_image(heatmap, original_data=input_data[0], show_plot=False)
