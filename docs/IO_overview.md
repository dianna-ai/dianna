# Overview of methods I/O types and dimensions

This document provides an overview of the input and output of
functions/methods used in Dianna. There are two views to facilitate the
explorations:

- Dianna functions view 1: the view is based on the type of the data:
  Timeseries, Image, Text, Tabular
- Dianna functions view 2: the view is based on the methods: Explainer, RISE,
  LIME, KERNELSHAP, Visualization

Each view provides links to related parts in the documentation API.

## Dianna functions view 1

### Timeseries

| Input                     | Name                   | Type          | Dims/shape |
| :-------------------      | :--------------------: | :-----------: | :----:     |
| [explain_timeseries]      |  input_timeseries      | np.ndarray    |            |
| [rise_timeseries]         |  input_timeseries      | np.ndarray    |            |
| [lime_timeseries]         |  input_timeseries      | np.ndarray    | [batch_size, sequence_length, num_features] |
| [visualization.timeseries]|  x,y                   | np.ndarray    | x shape (number of time_steps), y shape (number_of_time_steps, number_of_channels) |


| Output                    | Name                    | Type         | value range |
| :-------------------      | :---------------------: | :-----------:| :----:      |
| [explain_timeseries]      |  heatmap per class      | np.ndarray   |             |
| [rise_timeseries]         |  heatmap per class      | np.ndarray   |[normalize] is applied   |
| [lime_timeseries]         |  an explanation object  | np.ndarray   |   scores    |
| [visualization.timeseries]|  figure                 | plt.Figure   |   -         |


### Image

| Input                | Name               | Type               | Dims/shape    |
| :------------------- | :----------------: | :----------------: | :----------:  |
| [explain_image]      |  input_image       | np.ndarray         |               |
| [rise_image]         |  input_image       | np.ndarray         |  RGB          |
| [lime_image]         |  input_data        | np.ndarray         |  RGB          |
| [kernelshap_image]   |  image             | np.ndarray         |  RGB          |
| [visualization.image]|  heatmap           |      -             |               |


| Output               | Name                 | Type                     | value range           |
| :------------------- | :------------------: | :---------------------:  | :-----------:         |
| [explain_image]      |  heatmap per class   | np.ndarray               |                       |
| [rise_image]         |  heatmap per class   | np.ndarray               |[normalize] is applied |
| [lime_image]         |  list of heatmaps    | np.ndarray               |  scores               |
| [kernelshap_image]   |  Explanation heatmap | np.ndarray               |   -                   |
| [visualization.image]|  plot (None)         | matplotlib.figure.Figure | defined by cmap       |


### Text

| Input                | Name               | Type          | Dims/shape |
| :------------------- | :----------------: | :-----------: | :-------:  |
| [explain_text]       |  input_text        | string        |            |
| [rise_text]          |  input_text        | np.ndarray    |            |
| [lime_text]          |  input_text        | np.ndarray    |            |
| [visualization.text] |  explanation       | list of tuples|            |


| Output               | Name                       | Type           | value range                      |
| :------------------- | :------------------------: | :-----------:  | :---------------:                |
| [explain_text]       |  (word, index, importance) | List of tuples |                                  |
| [rise_text]          |  heatmap per class         | np.ndarray     |[normalize] is applied            |
| [lime_text]          |  (word, index, importance) | List of tuples |    -                             |
| [visualization.text] |  plot (None)               | IPython.display| does normalize to max importance |


### Tabular

| Input                   | Name          | Type                  | Dims/shape |
| :-------------------    | :------------:| :------------------:  | :-------:  |
| [explain_tabular]       |  input_tabular| np.ndarray            |            |
| [lime_tabular]          |  input_tabular| np.ndarray            |            |
| [kernelshap_tabular]    |  input_tabular| np.ndarray            |            |
| [visualization.tabular] |  x,y          | np.ndarray, List(str) |            |


| Output                  | Name                    | Type                     | value range |
| :-------------------    | :---------------------: | :----------------:       | :--------:  |
| [explain_tabular]       |  heatmap per class      | 2D array                 |             |
| [lime_tabular]          |  an explanation object  | np.array                 |  -          |
| [kernelshap_tabular]    |  an explanation object  | np.array                 |  -          |
| [visualization.tabular] |  plot                   | matplotlib.pyplot.Figure |             |


## Dianna functions view 2

### Explainer

| Input                | Name               | Type          | Dims/shape |
| :------------------- | :----------------: | :-----------: | :----:     |
| [explain_timeseries] |  input_timeseries  | np.ndarray    |            |
| [explain_image]      |  input_image       | np.ndarray    | RGB        |
| [explain_text]       |  input_text        | string        |            |
| [explain_tabular]    |  input_tabular     | np.ndarray    |            |


| Output               | Name                       | Type           | value range |
| :------------------- | :------------------------: | :-----------:  | :----:      |
| [explain_timeseries] |  heatmap per class         | np.ndarray     |             |
| [explain_image]      |  heatmap per class         | 2D array       |             |
| [explain_text]       |  (word, index, importance) | List of tuples |             |
| [explain_tabular]    |  heatmap per class         | 2D array       |             |


### RISE

| Input                | Name               | Type          | Dims/shape |
| :------------------- | :----------------: | :-----------: | :----:     |
| [rise_timeseries]    |  input_timeseries  | np.ndarray    |            |
| [rise_image]         |  input_image       | np.ndarray    |            |
| [rise_text]          |  input_text        | np.ndarray    |            |


| Output               | Name                       | Type           | value range           |
| :------------------- | :------------------------: | :-----------:  | :----------:          |
| [rise_timeseries]    |  heatmap per class         | np.ndarray     |[normalize] is applied |
| [rise_image]         |  heatmap per class         | np.ndarray     |[normalize] is applied |
| [rise_text]          |  heatmap per class         | np.ndarray     |[normalize] is applied |


### LIME

| Input                | Name               | Type          | Dims/shape |
| :------------------- | :----------------: | :-----------: | :----:     |
| [lime_timeseries]    |  input_timeseries  | np.ndarray    | [batch_size, sequence_length, num_features] |
| [lime_image]         |  input_data        | np.ndarray    |  RGB    |
| [lime_text]          |  input_text        | np.ndarray    |         |
| [lime_tabular]       |  input_tabular     | np.ndarray    |         |


| Output               | Name                       | Type           | value range |
| :------------------- | :------------------------: | :-----------:  | :--------:  |
| [lime_timeseries]    |  an explanation object     | (np.ndarray)*  |   scores    |
| [lime_image]         |  list of heatmaps          | list           |             |
| [lime_text]          |  (word, index, importance) | List of tuples |             |
| [lime_tabular]       |  an explanation object     | np.array       |             |

`*` mismatch between API doc and implementation

### KERNELSHAP

| Input                | Name               | Type          | Dims/shape |
| :------------------- | :----------------: | :-----------: | :-------:  |
| [kernelshap_image]   |  image             | np.ndarray    |            |
| [kernelshap_tabular] |  input_tabular     | np.ndarray    |            |


| Output               | Name                       | Type                | value range |
| :------------------- | :------------------------: | :----------------:  | :-------:   |
| [kernelshap_image]   |  Explanation heatmap       | np.ndarray (tuple)* |             |
| [kernelshap_tabular] |  an explanation object     | np.array            |             |

`*` mismatch between API doc and implementation

### Visualization

| Input                     | Name               | Type                  | Dims/shape |
| :-------------------      | :----------------: | :------------------:  | :----:     |
| [visualization.timeseries]|  x,y               | np.ndarray            | x shape (number of time_steps), y shape (number_of_time_steps, number_of_channels) |
| [visualization.image]     |  heatmap           |                       |            |
| [visualization.text]      |  explanation       | list of tuples        |            |
| [visualization.tabular]   |  x,y               | np.ndarray, List(str) |            |


| Output                    | Name         | Type                     | value range |
| :-------------------      | :----------: | :-------------------:    | :-------:   |
| [visualization.timeseries]|  figure      | plt.Figure               |             |
| [visualization.image]     |  plot (None) | matplotlib.figure.Figure |             |
| [visualization.text]      |  plot (None) | IPython.display          | does normalize to max importance|
| [visualization.tabular]   |  plot        | matplotlib.figure.Figure |             |



<!-- Links -->

[explain_timeseries]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/index.html#dianna.explain_timeseries
[explain_image]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/index.html#dianna.explain_image
[explain_text]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/index.html#dianna.explain_text
[explain_tabular]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/index.html#dianna.explain_tabular

[rise_timeseries]: https://dianna.readthedocs.io/en/latest/autoapi/rise_timeseries/index.html#rise_timeseries.RISETimeseries
[rise_image]: https://dianna.readthedocs.io/en/latest/autoapi/rise_image/index.html#rise_image.RISEImage
[rise_text]: https://dianna.readthedocs.io/en/latest/autoapi/rise_text/index.html#rise_text.RISEText

[lime_timeseries]: https://dianna.readthedocs.io/en/latest/autoapi/lime_timeseries/index.html#lime_timeseries.LIMETimeseries
[lime_image]: https://dianna.readthedocs.io/en/latest/autoapi/lime_image/index.html#lime_image.LIMEImage
[lime_text]: https://dianna.readthedocs.io/en/latest/autoapi/lime_text/index.html#lime_text.LIMEText
[lime_tabular]: https://dianna.readthedocs.io/en/latest/autoapi/lime_tabular/index.html#lime_tabular.LIMETabular

[kernelshap_image]: https://dianna.readthedocs.io/en/latest/autoapi/kernelshap_image/index.html#kernelshap_image.KERNELSHAPImage
[kernelshap_tabular]: https://dianna.readthedocs.io/en/latest/autoapi/kernelshap_tabular/index.html#kernelshap_tabular.KERNELSHAPTabular

[visualization.timeseries]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/visualization/timeseries/index.html#module-dianna.visualization.timeseries
[visualization.image]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/visualization/image/index.html#module-dianna.visualization.image
[visualization.text]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/visualization/text/index.html#module-dianna.visualization.text
[visualization.tabular]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/visualization/tabular/index.html#module-dianna.visualization.tabular

[normalize]: https://dianna.readthedocs.io/en/latest/autoapi/dianna/utils/rise_utils/index.html#dianna.utils.rise_utils.normalize
