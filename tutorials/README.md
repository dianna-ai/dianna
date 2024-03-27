<img width="150" alt="Logo_ER10" src="https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png">

## Tutorials
This folder contains DIANNA tutorial notebooks.

To install the dependencies for the tutorials, run
```
pip install .[notebooks]
```

For *general demonstration of DIANNA* click on the logo [<img width="75" alt="Logo_ER10" src="https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png">](https://github.com/dianna-ai/dianna/blob/main/tutorials/demo.ipynb) or run it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/demo.ipynb).

For *tutorials on how to convert* an [Keras](https://keras.io/), [PyTorch](https://pytorch.org/), [Scikit-learn](https://scikit-learn.org/stable/) or [Tensorflow](https://www.tensorflow.org/) model to [ONNX](https://onnx.ai/), please see the [conversion tutorials](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/).

For *specific XAI methods (explainers)*:
* Click on the explainer names to watch explanatory videos for the respective method.
* Click on the logos below for direct access to a tutorial notebook for a combination of explainer and data modality/dataset.

Run the tutorials directly in Google Colab by clicking on the Colab buttons below:

|*Modality* \ Method|RISE|[LIME](https://youtu.be/d6j6bofhj2M)|Kernel[SHAP](https://youtu.be/9haIOplEIGM)|
|:-----|:---|:---|:---|
|*Images*|[<img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152540187-b7a8239f-6742-437f-8f9b-35b950ce5ddb.png">](./explainers/RISE/rise_mnist.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/RISE/rise_mnist.ipynb) | [<img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/151539100-dbdfe0f8-485f-45d4-a249-a1f79e970066.png">](./explainers/LIME/lime_images.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/LIME/lime_images.ipynb) | [<img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152540187-b7a8239f-6742-437f-8f9b-35b950ce5ddb.png">](./explainers/KernelSHAP/kernelshap_mnist.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/KernelSHAP/kernelshap_mnist.ipynb) |
| | [<img width="94" alt="ImageNet_autocrop" src="https://user-images.githubusercontent.com/3244249/152542090-fd78fde1-6dec-43b6-a7ae-eea964b8ae28.png">](./explainers/RISE/rise_imagenet.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/RISE/rise_imagenet.ipynb) | | [<img width="20" alt="SimpleGeometric Logo" src="https://user-images.githubusercontent.com/3244249/151539027-f2fc3fc0-282a-4993-9680-74ee28bcd360.png">](./explainers/KernelSHAP/kernelshap_geometric_shapes.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/KernelSHAP/kernelshap_geometric_shapes.ipynb)|
|*Text* |[<img width="25" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152540890-c8e1e37d-f0cc-4f84-80a4-2c59176cbf4c.png">](./explainers/RISE/rise_text.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/RISE/rise_text.ipynb) |[<img width="25" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152540890-c8e1e37d-f0cc-4f84-80a4-2c59176cbf4c.png">](./explainers/LIME/lime_text.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/LIME/lime_text.ipynb)  |[]()|
| *Time series*| [<img width="25" alt="Weather Logo" src="https://user-images.githubusercontent.com/3244249/242001499-3ff3d639-ed2f-4a38-b7ac-957c984bce9f.png">](./explainers/RISE/rise_timeseries_weather.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/RISE/rise_timeseries_weather.ipynb)| [<img width="25" alt="Weather Logo" src="https://user-images.githubusercontent.com/3244249/242001499-3ff3d639-ed2f-4a38-b7ac-957c984bce9f.png">](./explainers/LIME/lime_timeseries_weather.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/LIME/lime_timeseries_weather.ipynb)| |
| | [<img width="25" alt="FRB logo" src="https://github.com/dianna-ai/dianna/assets/6370787/f53b280d-94b0-40ec-bfe7-ee48777d7964">](./explainers/RISE/rise_timeseries_frb.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/RISE/rise_timeseries_frb.ipynb) | [<img width="25" alt="Coffee Logo" src="https://user-images.githubusercontent.com/3244249/241999275-9ab50a0f-5da3-41d2-80e9-70d2c8769162.jpg">](./explainers/LIME/lime_timeseries_coffee.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/LIME/lime_timeseries_coffee.ipynb) | |
| *Tabular* | | [<img width="75" alt="Penguin Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/c7716ad3-f992-4557-80d9-1d8178c7ed57">](./explainers/LIME/lime_tabular_penguin.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/LIME/lime_tabular_penguin.ipynb) | |
| | | [<img width="25" alt="Weather Logo" src="https://user-images.githubusercontent.com/3244249/242001499-3ff3d639-ed2f-4a38-b7ac-957c984bce9f.png">](./explainers/LIME/lime_tabular_weather.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/explainers/LIME/lime_tabular_weather.ipynb)| |

The datasets used in the tutorials are represented with their respective logos:
|Data modality|Dataset|Logo|
|:------------|:------|:---|
|*Images*|Binary MNIST | <img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152540187-b7a8239f-6742-437f-8f9b-35b950ce5ddb.png">|
||[Simple Geometric (circles and triangles)](https://doi.org/10.5281/zenodo.5012824)| <img width="20" alt="SimpleGeometric Logo" src="https://user-images.githubusercontent.com/3244249/151539027-f2fc3fc0-282a-4993-9680-74ee28bcd360.png">|
||[Simple Scientific (LeafSnap30)](https://zenodo.org/record/5061353/)| <img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/151539100-dbdfe0f8-485f-45d4-a249-a1f79e970066.png"> |
||[Imagenet](https://image-net.org/download.php) | <img width="94" alt="ImageNet_autocrop" src="https://user-images.githubusercontent.com/3244249/152542090-fd78fde1-6dec-43b6-a7ae-eea964b8ae28.png">|
|*Text*| [Stanford sentiment treebank](https://nlp.stanford.edu/sentiment/index.html) | <img width="25" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152540890-c8e1e37d-f0cc-4f84-80a4-2c59176cbf4c.png">|
|*Timeseries* | [Coffee dataset](https://timeseriesclassification.com/description.php?Dataset=Coffee)  | <img width="25" alt="Coffe Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/9ab50a0f-5da3-41d2-80e9-70d2c8769162">|
|           | [Weather dataset](https://zenodo.org/record/7525955) | <img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f">|
|           | Fast Radio Burst (FRB) dataset (not publicly available) | <img width="25" alt="FRB logo" src="https://github.com/dianna-ai/dianna/assets/6370787/f53b280d-94b0-40ec-bfe7-ee48777d7964">|
|*Tabular*| [Penguin dataset](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris)| <img width="75" alt="Penguin Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/c7716ad3-f992-4557-80d9-1d8178c7ed57"> | |
|           | [Weather dataset](https://zenodo.org/record/7525955) | <img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f">|

The ONNX models used in the tutorials are available at [dianna/models](https://github.com/dianna-ai/dianna/tree/main/dianna/models), or linked from their respective tutorial notebooks.

### IMPORTANT: Hyperparameters
The XAI methods (explainers) are sensitive to the choice of their hyperparameters! In this [work](https://staff.fnwi.uva.nl/a.s.z.belloum/MSctheses/MScthesis_Willem_van_der_Spec.pdf), this sensitivity is researched and useful conclusions are drawn.
The default hyperparameters used in DIANNA for each explainer as well as the choices for some tutorials and their data modality (*i* - images, *txt* - text, *ts* - time series and *tab* - tabular) are given below:

#### RISE
| Hyperparameter  | Default value | <img width="94" alt="ImageNet_autocrop" src="https://user-images.githubusercontent.com/3244249/152542090-fd78fde1-6dec-43b6-a7ae-eea964b8ae28.png"> (*i*)| <img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152540187-b7a8239f-6742-437f-8f9b-35b950ce5ddb.png">(*i*) | <img width="25" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152540890-c8e1e37d-f0cc-4f84-80a4-2c59176cbf4c.png"> (*txt*) | <img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f">  (*ts*)| <img width="25" alt="FRB logo" src="https://github.com/dianna-ai/dianna/assets/6370787/f53b280d-94b0-40ec-bfe7-ee48777d7964"> (*ts*)|
| ------------- | ------------- | -------------------|-----------------------------| ---------------------------------|---------------------------------|---------------------------------|
| $n_{masks}$  |**$1000$**  | default | $5000$ | default | $10000$ |$5000$ |
| $p_{keep}$  | **optimized** (*i*, *txt*), **$0.5$** (*ts*) | $0.1$| $0.1$ |  default | $0.1$| $0.1$|
| Resolution  |**$8$** | $6$ |default |  default | default | $1$ |
#### LIME
| Hyperparameter  | Value(s) |
| ------------- | ------------- |
| $n_{samples}$  | $1800$ (text), $1600$ (image)  |
| Kernel Width | $25$ (image and text)  |
| $n_{segments}$ | $95$ |
| L2 Regularization | $0.01$ |
#### KernalSHAP
| Hyperparameter  | Value(s) |
| ------------- | ------------- |
| $n_{samples}$  | $1600$ |
| $n_{segments}$ | $95$ |
| L1 Regularization | auto |
