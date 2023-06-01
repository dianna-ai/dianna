<img width="150" alt="Logo_ER10" src="https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png">

## Tutorials
This folder contains DIANNA tutorial notebooks. 

To install the dependencies for the tutorials, run
```
pip install .[notebooks]
```

For general demonstration of DIANNA click on the logo [<img width="75" alt="Logo_ER10" src="https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png">](https://github.com/dianna-ai/dianna/blob/main/tutorials/demo.ipynb) or run it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/demo.ipynb).

Click on the XAI method (explainer) names to watch explanatory videos for the respective method.

Use the clickable logos below for direct access to a tutorial notebook for an explainability method and data modality/dataset. 

The tutorials can also be run directly in Google Colab, by clicking on the Colab buttons below: 

|*Modality*\ Method|RISE|[LIME](https://youtu.be/d6j6bofhj2M)|Kernel[SHAP](https://youtu.be/9haIOplEIGM)|
|:-----|:---|:---|:---|
|*Images*|[<img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152540187-b7a8239f-6742-437f-8f9b-35b950ce5ddb.png">](rise_mnist.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/rise_mnist.ipynb) | [<img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/151539100-dbdfe0f8-485f-45d4-a249-a1f79e970066.png">](lime_images.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/lime_images.ipynb) | [<img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152540187-b7a8239f-6742-437f-8f9b-35b950ce5ddb.png">](kernelshap_mnist.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/kernelshap_mnist.ipynb) |
| | [<img width="94" alt="ImageNet_autocrop" src="https://user-images.githubusercontent.com/3244249/152542090-fd78fde1-6dec-43b6-a7ae-eea964b8ae28.png">](rise_imagenet.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/rise_imagenet.ipynb) | | [<img width="20" alt="SimpleGeometric Logo" src="https://user-images.githubusercontent.com/3244249/151539027-f2fc3fc0-282a-4993-9680-74ee28bcd360.png">](kernelshap_geometric_shapes.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/kernelshap_geometric_shapes.ipynb)|
|*Text* |[<img width="25" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152540890-c8e1e37d-f0cc-4f84-80a4-2c59176cbf4c.png">](rise_text.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/rise_text.ipynb) |[<img width="25" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152540890-c8e1e37d-f0cc-4f84-80a4-2c59176cbf4c.png">](lime_text.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/lime_text.ipynb)  |[]()|
| *Time series*| [<img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f">](https://github.com/dianna-ai/dianna/blob/main/tutorials/rise_timeseries_weather.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/rise_timeseries_weather.ipynb)| [<img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f">](https://github.com/dianna-ai/dianna/blob/main/tutorials/lime_timeseries_weather.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/lime_timeseries_weather.ipynb)| |
| | | [<img width="25" alt="Coffe Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/9ab50a0f-5da3-41d2-80e9-70d2c8769162">](https://github.com/dianna-ai/dianna/blob/main/tutorials/lime_timeseries_coffee.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/lime_timeseries_coffee.ipynb) | |

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

The ONNX models used in the tutorials are available at [tutorials/models](https://github.com/dianna-ai/dianna/tree/main/tutorials/models).



