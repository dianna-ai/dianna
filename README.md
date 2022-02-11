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
affiliations:
 - name: Netherlands eScience Center, Amsterdam, the Netherlands
   index: 1
-->

[![build](https://github.com/dianna-ai/dianna/actions/workflows/build.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=coverage)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5542/badge)](https://bestpractices.coreinfrastructure.org/projects/5542)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)

<img width="150" alt="Logo_ER10" src="https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png">

<img align="left" src="https://user-images.githubusercontent.com/55382553/153408200-36c4de2e-7865-4934-956d-09eefd893e6a.png">

# Deep Insight And Neural Network Analysis

DIANNA is a Python package that brings explainable AI (XAI) to your research project. It wraps carefully selected XAI methods in a simple, uniform interface.
It's built by, with and for (academic) researchers and research software engineers working on machine learning projects.

## Why DIANNA? 
DIANNA software is addressing needs of both (X)AI reseachers and mostly the various domains scientists who are using or will use AI models for their research without being experts in (X)AI. DIANNA is future-proof: the only XAI library supporting the [Open Neural Network Exchange (ONNX)](https://onnx.ai/) format. 

* Provides an easy-to-use interface for non (X)AI experts
* Implements well-known XAI methods (LIME, RISE and Kernal SHAP) chosen by systematic and objective evaluation criteria
* Supports the de-facto standard format for neural network models - ONNX.
* Includes clear instructions for export/conversions from Tensorflow, Pytorch, Keras and skikit-learn to ONNX.
* Supports both images and text data modalities. Time series, tabular data and even embeddings support is planned.
* Comes with simple intuitive image and text benchmarks 
* Easily extendable to other XAI methods 

For more information on the unique stengths of DIANNA with comparision to other tools, please see the [context landscape](https://dianna.readthedocs.io/en/latest/CONTEXT.html).

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

### Pre-requisites only for Macbook Pro with M1 Pro chip users

- To install TensorFlow you can follow this [tutorial](https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/).

- To install TensorFlow Addons you can follow these [steps](https://github.com/tensorflow/addons/pull/2504). For further reading see this [issue](https://github.com/tensorflow/addons/issues/2503). 

## How to use DIANNA
To use DIANNA you need a _trained AI model_ (in ONNX format) and a _data item_ (e.g. an image or text, etc.) for which you would like to explain the output of the model. 
DIANNA calls an explainable AI method to produce the relevance scores of each data pont (e.g. pixel, word) to a given model's decision overlaid on the data item. 

For example usage see the DIANNA [tutorials](./tutorials). For creating or converting a trained model to ONNX see the **ONNX models** and for example datasets- the **Datasets**  sections below.

![Architecture_high_level_resized](https://user-images.githubusercontent.com/3244249/152557189-3ed6fe1a-b461-4cc8-bd2e-e420ee46c784.png)



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

## ONNX models
<!-- TODO: Add all links, see issue https://github.com/dianna-ai/dianna/issues/135 -->

**We work with ONNX!** ONNX is a great unified neural network standard which can be used to boost reproducible science. Using ONNX for your model also gives you a boost in performance! In case your models are still in another popular DNN (deep neural network) format, here are some simple recipes to convert them:
* [pytorch](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/pytorch2onnx.ipynb) - use the built-in [`torch.onnx.export`](https://pytorch.org/docs/stable/onnx.html) function to convert pytorch models to onnx.
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

**_We envision the birth of the ONNX Scientific models zoo soon..._**

## Tutorials
DIANNA supports different data modalities and XAI methods. The table contains links to the relevant XAI method's papers. There are DIANNA [tutorials](./tutorials) covering each supported method and data modality on a least one dataset. Our future plans to expand DIANNA with more data modalities and XAI methods are given in the [ROADMAP](https://dianna.readthedocs.io/en/latest/ROADMAP.html).

<!-- see issue: https://github.com/dianna-ai/dianna/issues/142, also related issue: https://github.com/dianna-ai/dianna/issues/148 -->

|Data \ XAI|[RISE](http://bmvc2018.org/contents/papers/1064.pdf)|[LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)|[KernelSHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)|
|:-----|:---|:---|:---|
|Images|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Text|:white_check_mark:|:white_check_mark:|planned|
|Embedding|coming soon|coming soon|coming soon|
|Timeseries|planned|planned|planned|
|Tabular|planned|planned|planned|
|Graphs | | | |

[LRP](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable) and [PatternAttribution](https://arxiv.org/pdf/1705.05598.pdf) also feature in the top 5 of our thoroughly evaluated XAI methods using objective critera (details in coming blog-post). **Contributing by adding these and more (new) post-hoc explainability methods on ONNX models is very welcome!**

## Reference documentation 

For detailed information on using specific DIANNA functions, please visit the [documentation page hosted at Readthedocs](https://dianna.readthedocs.io/en/latest).

## Contributing

If you want to contribute to the development of DIANNA,
have a look at the [contribution guidelines](https://dianna.readthedocs.io/en/latest/CONTRIBUTING.html).

## How to cite us 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592607.svg)](https://zenodo.org/record/5592607)
[![RSD](https://img.shields.io/badge/rsd-dianna-00a3e3.svg)](https://www.research-software.nl/software/dianna)

If you use this package for your scientific work, please consider citing it as:

    Ranguelova, Elena, Bos, Patrick, Liu, Yang, Meijer, Christiaan, & Oostrum, Leon. (2021). dianna (*[VERSION YOU USED]*). Zenodo. https://zenodo.org/record/5592607

See also the [Zenodo page](https://zenodo.org/record/5592607) for exporting the citation to BibTteX and other formats.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
