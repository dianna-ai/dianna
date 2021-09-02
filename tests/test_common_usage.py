import numpy as np
import dianna
import dianna.visualization


model = "Some model"
input_data = np.ones((5, 3))


def test_common_SHAP_pipeline():  # noqa: N802 ignore case
    heatmap = dianna.explain(model, input_data, method="SHAP")
    dianna.visualization.plot_image(heatmap, show_plot=False)
    dianna.visualization.plot_image(heatmap, original_data=input_data, show_plot=False)
