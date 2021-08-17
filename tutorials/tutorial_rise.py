import numpy as np
import dianna
import dianna.visualization


def run_model(input_data):
    n_class = 2
    batch_size = input_data.shape[0]

    return np.random.random((batch_size, n_class))


# shape is batch, y, x, channel
input_data = np.random.random((1, 224, 224, 3))

heatmaps = dianna.explain(run_model, input_data, method="RISE", n_masks=200)
# output is one heatmap per class
dianna.visualization.plot_image(heatmaps[0])
dianna.visualization.plot_image(heatmaps[0], original_data=input_data[0])
