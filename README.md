<!--
title: 'DIANNA: Deep Insight And Neural Network Analysis'
tags:
  - Python
  - explainable AI
  - deep neural networks
  - ONNX
  - benchmark sets
authors:
  - name: Elena Ranguelova^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-9834-1756
    affiliation: 1
  - name: Patrick Bos^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-6033-960X
    affiliation: 1
  - name: Yang Liu^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-1966-8460
    affiliation: 1
  - name: Christiaan Meijer^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-5529-5761
    affiliation: 1
  - name: Leon Oostrum^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0001-8724-8372
    affiliation: 1
  - name: Giulia Crocioni^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-0823-0121
    affiliation: 1
  - name: Laura Ootes^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-2800-8309
    affiliation: 1  
  - name: Pranav Chandramouli^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-7896-2969
    affiliation: 1  
  - name: Aron Jansen^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-4764-9347
    affiliation: 1  
  - name: Stef Smeets^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-5413-9038
    affiliation: 1  
affiliations:
 - name: Netherlands eScience Center, Amsterdam, the Netherlands
   index: 1
-->

[![build](https://github.com/dianna-ai/dianna/actions/workflows/build.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=coverage)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5542/badge)](https://bestpractices.coreinfrastructure.org/projects/5542)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
 [![status](https://camo.githubusercontent.com/dcc6405df4084ef5aa1cdf0f13d7fc01e72c9e7c4ca907a68c95698cec85e75a/68747470733a2f2f6a6f73732e7468656f6a2e6f72672f7061706572732f66303539326331616563623337313165303638623538393730353838663138352f7374617475732e737667)](https://joss.theoj.org/papers/f0592c1aecb3711e068b58970588f185)

<img width="300" alt="Logo_ER10" src="https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png">

<img align="left" src="https://user-images.githubusercontent.com/55382553/153408200-36c4de2e-7865-4934-956d-09eefd893e6a.png">

# Deep Insight And Neural Network Analysis

DIANNA is a Python package that brings explainable AI (XAI) to your research project. It wraps carefully selected XAI methods in a simple, uniform interface.
It's built by, with and for (academic) researchers and research software engineers working on machine learning projects.

## Why DIANNA?
DIANNA software is addressing needs of both (X)AI researchers and mostly the various domains scientists who are using or will use AI models for their research without being experts in (X)AI. DIANNA is future-proof: one of the very few XAI library supporting the [Open Neural Network Exchange (ONNX)](https://onnx.ai/) format.

After studying the vast XAI landscape we have made choices in the parts of the [XAI Taxonomy](https://doi.org/10.3390/make3030032) on which methods, data modalities and problems types to focus. Our choices, based on the largest usage in scientific literature, are shown graphically in the XAI taxonomy below:

<img src="https://user-images.githubusercontent.com/33344129/234316105-0bc5721f-c4b0-432d-88a7-7586f034ec2c.png" alt="XAI_taxonomy" width="60%"/>

The key points of DIANNA:

* Provides an easy-to-use interface for non (X)AI experts
* Implements well-known XAI methods (LIME, RISE and Kernal SHAP) chosen by systematic and objective evaluation criteria
* Supports the de-facto standard format for neural network models - ONNX.
* Includes clear instructions for export/conversions from Tensorflow, Pytorch, Keras and scikit-learn to ONNX.
* Supports images, text and time series data modalities. Tabular data and even embeddings support is planned.
* Comes with simple intuitive image and text benchmarks
* Easily extendable to other XAI methods

For more information on the unique strengths of DIANNA with comparison to other tools, please see the [context landscape](https://dianna.readthedocs.io/en/latest/CONTEXT.html).

## Installation
[![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/)
[![supported python versions](https://img.shields.io/pypi/pyversions/dianna)](https://pypi.python.org/project/dianna/)

DIANNA can be installed from PyPI using [pip](https://pip.pypa.io/en/stable/installation/) on any of the supported Python versions (see badge):

```console
python3 -m pip install dianna
```

To install the most recent development version directly from the GitHub repository run:

```console
python3 -m pip install git+https://github.com/dianna-ai/dianna.git
```

If you get an error related to OpenMP when importing dianna, have a look at [this issue](https://github.com/dianna-ai/dianna/issues/376) for possible workarounds.

### Pre-requisites only for Macbook Pro with M1 Pro chip users

- To install TensorFlow you can follow this [tutorial](https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/).

- To install TensorFlow Addons you can follow these [steps](https://github.com/tensorflow/addons/pull/2504). For further reading see this [issue](https://github.com/tensorflow/addons/issues/2503). Note that this temporary solution works only for macOS versions >= 12.0. Note that this step may have changed already, see https://github.com/dianna-ai/dianna/issues/245.

- Before installing DIANNA, comment `tensorflow` requirement in `setup.cfg` file (tensorflow package for M1 is called `tensorflow-macos`).

## Getting started
You need:
- your trained ONNX model ([convert my pytorch/tensorflow/keras/scikit-learn model to ONNX](https://github.com/dianna-ai/dianna#onnx-models))
- 1 data item to be explained

 You get:
 - a relevance map overlayed over the data item

In the library's documentation, the general usage is explained in [How to use DIANNA](https://dianna.readthedocs.io/en/latest/usage.html)

### Demo movie

[![Watch the video on YouTube](https://img.youtube.com/vi/u9_c5DJewLU/default.jpg)](https://youtu.be/u9_c5DJewLU)

### Text example:
```python
model_path = 'your_model.onnx'  # model trained on text
text = 'The movie started great but the ending is boring and unoriginal.'
```
Which of your model's classes do you want an explanation for?
```python
labels = [positive_class, negative_class]
```
Run using the XAI method of your choice, for example LIME:
```python
explanation = dianna.explain_text(model_path, text, 'LIME')
dianna.visualization.highlight_text(explanation[labels.index(positive_class)], text)
```
![image](https://user-images.githubusercontent.com/6087314/155532504-6f90f032-cbb4-4e71-9b99-aa9c0de4e86a.png)


### Image example:
```python
model_path = 'your_model.onnx'  # model trained on images
image = PIL.Image.open('your_image.jpeg')
```
Tell us what label refers to the channels, or colors, in the image.
```python
axis_labels = {0: 'channels'}
```
Which of your model's classes do you want an explanation for?
```python
labels = [class_a, class_b]
```
Run using the XAI method of your choice, for example RISE:
```python
explanation = dianna.explain_image(model_path, image, 'RISE', axis_labels=axis_labels, labels=labels)
dianna.visualization.plot_image(explanation[labels.index(class_a)], original_data=image)
```
![image](https://user-images.githubusercontent.com/6087314/155557077-e2052094-d8ac-49d3-a840-0160256d53a6.png)

## Dashboard

Explore your trained model explained using the DIANNA dashboard. [Click here](https://github.com/dianna-ai/dianna/tree/main/dianna/dashboard) for more information.

<a href="https://github.com/dianna-ai/dianna/tree/main/dianna/dashboard" target="_blank">
  <img width="1000" align="center" alt="Dianna dashboard screenshot" src="https://raw.githubusercontent.com/dianna-ai/dianna/main/dianna/dashboard/dashboard-screenshot.png">
</a>

## Datasets
DIANNA comes with simple datasets. Their main goal is to provide intuitive insight into the working of the XAI methods. They can be used as benchmarks for evaluation and comparison of existing and new XAI methods.

### Images
|Dataset|Description|Examples|Generation|
|:-----|:----|:---|:----|
|Binary MNIST <img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152354583-d7b68902-d402-4098-922b-b1a33b07e3e1.png">| Greyscale images of the digits "1" and "0" - a 2-class subset from the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for handwritten digit classification. |<img width="120" alt="BinaryMNIST" src="https://user-images.githubusercontent.com/3244249/150808267-3d27eae0-78f2-45f8-8569-cb2561f2c2e9.png">| [Binary MNIST dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/MNIST)|
|[Simple Geometric (circles and triangles)](https://doi.org/10.5281/zenodo.5012824) <img width="20" alt="Simple Geometric Logo" src="https://user-images.githubusercontent.com/3244249/150808842-d35d741e-294a-4ede-bbe9-58e859483589.png"> | Images of circles and triangles for 2-class geometric shape classificaiton. The shapes of varying size and orientation and the background have varying uniform gray levels.  | <img width="130" alt="SimpleGeometric" src="https://user-images.githubusercontent.com/3244249/150808125-e1576237-47fa-4e51-b01e-180904b7c7f6.png">| [Simple geometric shapes dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/geometric_shapes) |
|[Simple Scientific (LeafSnap30)](https://zenodo.org/record/5061353/)<img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/150815639-2da560d4-8b26-4eeb-9ab4-dabf221a264a.png"> | Color images of tree leaves - a 30-class post-processed subset from the LeafSnap dataset for automatic identification of North American tree species.|<img width="600" alt="LeafSnap" src="https://user-images.githubusercontent.com/3244249/150804246-f714e517-641d-48b2-af26-2f04166870d6.png">| [LeafSnap30 dataset generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/dataset_preparation/LeafSnap/)|

### Text
|Dataset|Description|Examples|Generation|
|:-----|:----|:---|:----|
| [Stanford sentiment treebank](https://nlp.stanford.edu/sentiment/index.html)<img width="20" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152355020-908c04f3-aa99-489d-b87a-7e6b1f586118.png">|Dataset for predicting the sentiment, positive or negative, of movie reviews. | _This movie was actually neither that funny, nor super witty._|[Sentiment treebank](https://nlp.stanford.edu/sentiment/treebank.html)|

### Time series
|Dataset|Description|Examples|Generation|
|:-----|:----|:---|:----|
| [Coffee dataset](https://timeseriesclassification.com/description.php?Dataset=Coffee) <img width="25" alt="Coffe Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/9ab50a0f-5da3-41d2-80e9-70d2c8769162"> | Food spectographs time series dataset for a two class problem to distinguish between Robusta and Arabica coffee beans.| <img width="500" alt="example image" src="https://github.com/dianna-ai/dianna/assets/3244249/763002c5-40ad-48cc-9de0-ea43d7fa8a75)">| [data source](https://github.com/QIBChemometrics/Benchtop-NMR-Coffee-Survey)| 
|  [Weather dataset](https://zenodo.org/record/7525955) <img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f"> |The light version of the weather prediciton dataset, which contains daily observations (89 features) for 11 European locations through the years 2000 to 2010.| <img width="500" alt="example image" src="https://github.com/dianna-ai/dianna/assets/3244249/b0a505ac-8a6c-4e1c-b6ad-35e31e52f46d)"> | [data source](https://github.com/florian-huber/weather_prediction_dataset) |

## ONNX models
<!-- TODO: Add all links, see issue https://github.com/dianna-ai/dianna/issues/135 -->

**We work with ONNX!** ONNX is a great unified neural network standard which can be used to boost reproducible science. Using ONNX for your model also gives you a boost in performance! In case your models are still in another popular DNN (deep neural network) format, here are some simple recipes to convert them:
* [pytorch and pytorch-lightning](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/pytorch2onnx.ipynb) - use the built-in [`torch.onnx.export`](https://pytorch.org/docs/stable/onnx.html) function to convert pytorch models to onnx, or call the built-in [`to_onnx`](https://pytorch-lightning.readthedocs.io/en/latest/deploy/production_advanced.html) function on your [`LightningModule`](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule) to export pytorch-lightning models to onnx.
* [tensorflow](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/tensorflow2onnx.ipynb) - use the [`tf2onnx`](https://github.com/onnx/tensorflow-onnx) package to convert tensorflow models to onnx.
* [keras](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/keras2onnx.ipynb) - same as the conversion from tensorflow to onnx, the [`tf2onnx`](https://github.com/onnx/tensorflow-onnx) package also supports keras.
* [scikit-learn](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/skl2onnx.ipynb) - use the [`skl2onnx`](https://github.com/onnx/sklearn-onnx) package to scikit-learn models to onnx.

More converters with examples and tutorials can be found on the [ONNX tutorial page](https://github.com/onnx/tutorials).

And here are links to notebooks showing how we created our models on the benchmark datasets:
### Images
|Models|Generation|
|:-----|:----|
|[Binary MNIST model](https://zenodo.org/record/5907177)| [Binary MNIST model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/MNIST/generate_model_binary.ipynb)|
|[Simple Geometric model](https://zenodo.org/deposit/5907059)| [Simple geometric shapes model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/geometric_shapes/generate_model.ipynb)|
|[Simple Scientific model](https://zenodo.org/record/5907196)| [LeafSnap30 model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/LeafSnap/generate_model.ipynb)|

### Text
|Models|Generation|
|:-----|:----|
|[Movie reviews model](https://zenodo.org/record/5910598)| [Stanford sentiment treebank model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/movie_reviews/generate_model.ipynb)|

### Time series
|Models|Generation|
|:-----|:----|
| Coffee model | [Coffee model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/coffee/generate_model.ipynb)|
| [Season prediction model](https://zenodo.org/record/7543883) | [Season prediction model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/season_prediction/generate_model.ipynb) |

**_We envision the birth of the ONNX Scientific models zoo soon..._**

## Tutorials
DIANNA supports different data modalities and XAI methods. The table contains links to the relevant XAI method's papers (for some explanatory videos on the methods, please see [tutorials](./tutorials)). The DIANNA [tutorials](./tutorials) cover each supported method and data modality on a least one dataset. Our future plans to expand DIANNA with more data modalities and XAI methods are given in the [ROADMAP](https://dianna.readthedocs.io/en/latest/ROADMAP.html).

<!-- see issue: https://github.com/dianna-ai/dianna/issues/142, also related issue: https://github.com/dianna-ai/dianna/issues/148 -->

|Data \ XAI|[RISE](http://bmvc2018.org/contents/papers/1064.pdf)|[LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)|[KernelSHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)|
|:-----|:---|:---|:---|
|Images|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Text|:white_check_mark:|:white_check_mark:||
|Timeseries|:white_check_mark:|:white_check_mark:||
|Embedding|planned|planned|planned|
|Tabular|planned|planned|planned|
|Graphs* |work in progress|work in progress| work in progress|



[LRP](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable) and [PatternAttribution](https://arxiv.org/pdf/1705.05598.pdf) also feature in the top 5 of our thoroughly evaluated XAI methods using objective criteria (details in coming blog-post). **Contributing by adding these and more (new) post-hoc explainability methods on ONNX models is very welcome!**

## Reference documentation

For detailed information on using specific DIANNA functions, please visit the [documentation page hosted at Readthedocs](https://dianna.readthedocs.io/en/latest).

## Contributing

If you want to contribute to the development of DIANNA,
have a look at the [contribution guidelines](https://dianna.readthedocs.io/en/latest/CONTRIBUTING.html).
See our [developer documentation](docs/developer_info.rst) for information on developer installation, running tests, generating documentation, versioning and making a release.

## How to cite us
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592606.svg)](https://zenodo.org/record/5592606)
[![RSD](https://img.shields.io/badge/rsd-dianna-00a3e3.svg)](https://www.research-software.nl/software/dianna)

If you use this package for your scientific work, please consider citing directly the software as:

    Ranguelova, E., Bos, P., Liu, Y., Meijer, C., Oostrum, L., Crocioni, G., Ootes, L., Chandramouli, P., Jansen, A., Smeets, S. (2023). dianna (*[VERSION YOU USED]*). Zenodo. https://zenodo.org/record/5592606

or the JOSS paper as:

    Ranguelova et al., (2022). DIANNA: Deep Insight And Neural Network Analysis. Journal of Open Source Software, 7(80), 4493, https://doi.org/10.21105/joss.04493

See also the [Zenodo page](https://zenodo.org/record/5592606) or the [JOSS page](https://joss.theoj.org/papers/10.21105/joss.04493) for exporting the software citation to BibTteX and other formats.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
