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
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04493/status.svg)](https://doi.org/10.21105/joss.04493)


<img width="300" alt="Logo_ER10" src="https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png">

[Introducing DIANNA video](https://youtu.be/9VM5acip2s8)

<img align="left" src="https://user-images.githubusercontent.com/55382553/153408200-36c4de2e-7865-4934-956d-09eefd893e6a.png">

# Deep Insight And Neural Network Analysis

DIANNA is a Python package that brings explainable AI (XAI) to your research project. It wraps carefully selected XAI methods in a simple, uniform interface.
It's built by, with and for (academic) researchers and research software engineers working on machine learning projects.

## Why DIANNA?

DIANNA software is addressing needs of both (X)AI researchers and mostly the various domains scientists who are using or will use AI models for their research without being experts in (X)AI. DIANNA is future-proof: one of the very few XAI library supporting the [Open Neural Network Exchange (ONNX)](https://onnx.ai/) format.

After studying the vast XAI landscape we have made choices in the parts of the [XAI Taxonomy](https://doi.org/10.3390/make3030032) on which methods, data modalities and problems types to focus. Our choices, based on the largest usage in scientific literature, are shown graphically in the XAI taxonomy below:

<img src="https://github.com/dianna-ai/dianna/assets/3244249/9b864980-86f4-4d0e-8a83-af7d6be606f7" alt="XAI_taxonomy" width="80%"/>

The key points of DIANNA:

 *   Provides an easy-to-use interface for non (X)AI experts
 *   Implements well-known XAI methods LIME, RISE and KernelSHAP, chosen by systematic and objective evaluation criteria
 *   Comes with a dashboard where  results of different explainers can be compared for all data types
 *   Supports the de-facto standard of neural network models - ONNX
 *   Supports images, text, time series, tabular data modalities and embeddings (in a related [package](https://github.com/dianna-ai/explainable_embedding))
 *   Comes with simple intuitive image, text, time series, and tabular benchmarks, so can help you with your XAI research
 *   Includes scientific use-cases tutorials
 *   Easily extendable to other XAI methods


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

<details><summary>Pre-requisites only for Macbook Pro with M1 Pro chip users</summary>
<p>
  
- To install TensorFlow you can follow this [tutorial](https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/).
- To install TensorFlow Addons you can follow these [steps](https://github.com/tensorflow/addons/pull/2504). For further reading see this [issue](https://github.com/tensorflow/addons/issues/2503). Note that this temporary solution works only for macOS versions >= 12.0. Note that this step may have changed already, see https://github.com/dianna-ai/dianna/issues/245.
- Before installing DIANNA, comment `tensorflow` requirement in `setup.cfg` file (tensorflow package for M1 is called `tensorflow-macos`).
</details>

## Getting started

You need:

- your trained ONNX model ([convert my pytorch/tensorflow/keras/scikit-learn model to ONNX](https://github.com/dianna-ai/dianna#onnx-models))
- a data item to be explained

 You get:

- a relevance map overlayed over the data item

### Template example for any data modality and explainer

1. Provide your *trained model* and *data item* ( *text, image, time series or tabular* )

```python
model_path = 'your_model.onnx'  # model trained on your data modality
data_item = <data_item> # data item for which the model's prediction needs to be explained
```

2. If the task is classification: which are the *classes* your model has been trained for?

```python
labels = [class_a, class_b]   # example of binary classification labels
```
*Which* of these classes do you want an explanation for?
```python
explained_class_index = labels.index(<explained_class>)  # explained_class can be any of the labels
```

3. Run dianna with the *explainer* of your choice ( *'LIME', 'RISE' or 'KernalSHAP'*) and visualize the output:

```python
explanation = dianna.<explanation_function>(model_path, data_item, explainer)
dianna.visualization.<visualization_function>(explanation[explained_class_index], data_item)
```

### Text and image usage 
<details><summary>Examples</summary>
<p>
Lets illustrate the template above with *textual* data. The data item of interest is a sentence being (a part of) a movie review and the model has been trained to classify reviews into positive and negative sentiment classes.
We are intersted which words are contributing positively (red) and which - negatively (blue) towards the model's desicion to classify the review as positive and we would like to use the *LIME* explainer:

```python
model_path = 'your_text_model.onnx'
# also define a model runner here (details in dedicated notebook)
review = 'The movie started great but the ending is boring and unoriginal.'
labels = ["negative", "positive"]
explained_class_index = labels.index("positive")
explanation = dianna.explain_text(model_path, text, 'LIME')
dianna.visualization.highlight_text(explanation[explained_class_index], model_runner.tokenizer.tokenize(review))
```

![image](https://user-images.githubusercontent.com/6087314/155532504-6f90f032-cbb4-4e71-9b99-aa9c0de4e86a.png)

Here is another illustration on how to use dianna to explain which parts of a bee *image* contributied positively (red) or negativey (blue) towards a classifying the image as a *'bee'* using *RISE*.
The Imagenet model has been trained to distinguish between 1000 classes (specified in ```labels```).
For images, which are data of higher dimention compared to text, there are also some specifics to consider:

```python
model_path = 'your_image_model.onnx'
image = PIL.Image.open('your_bee_image.jpeg')
axis_labels = {2: 'channels'}
explained_class_index = labels.index('bee')
explanation = dianna.explain_image(model_path, image, 'RISE', axis_labels=axis_labels, labels=labels)
dianna.visualization.plot_image(explanation[explained_class_index], utils.img_to_array(image)/255., heatmap_cmap='bwr')
plt.show()
```
<img src="https://github.com/dianna-ai/dianna/assets/3244249/b03e4d4e-e3e8-4248-bf62-e3602b7f6d71" width="215" height="215">

And why would Imagenet think the same image would be a *garden spider*?
```python
explained_class_index = labels.index('garden_spider') # interested in the image being classified as a garden spider
explanation = dianna.explain_image(model_path, image, 'RISE', axis_labels=axis_labels, labels=labels)
dianna.visualization.plot_image(explanation[explained_class_index], utils.img_to_array(image)/255., heatmap_cmap='bwr')
plt.show()
```

<img src="https://github.com/dianna-ai/dianna/assets/3244249/e7623803-2369-40ad-b4ef-4a6ae4e902f1" width="215" height="215">

</details>

### Overview tutorial
There are **full working examples** on how to use the supported explainers and how to use dianna for **all supported data modalities** in our [overview tutorial](./tutorials/overview.ipynb).

### IMPORTANT: Sensitivity to hyperparameters
The explainers are sensitive to the choice of their hyperparameters! In this [work](https://staff.fnwi.uva.nl/a.s.z.belloum/MSctheses/MScthesis_Willem_van_der_Spec.pdf), this sensitivity to hyperparameters is researched and useful conclusions are drawn.
The default hyperparameters used in DIANNA for each explainer as well as the values for our tutorial examples are given in the Tutorials [README](./tutorials/README.md#important-hyperparameters).

### Introductory video
This video shows the main functionality of DIANNA and shows you how to use DIANNA also from its dashboard.

[Watch the video on YouTube](https://youtu.be/9VM5acip2s8)

## Dashboard

![image](https://github.com/user-attachments/assets/27d621d0-eaab-4640-9353-691ae7adb12b)


Explore the explanations of your trained model using the DIANNA dashboard.
[Click here](https://github.com/dianna-ai/dianna/tree/main/dianna/dashboard) for more information.



## Datasets

DIANNA comes with simple datasets. Their main goal is to provide intuitive insight into the working of the XAI methods. They can be used as benchmarks for evaluation and comparison of existing and new XAI methods.

<details><summary>Images</summary>

| Dataset                                                                                                                                                                                                                                     | Description                                                                                                                                                                 | Examples                                                                                                                                               | Generation                                                                                                                                             |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Binary MNIST <img width="25" alt="mnist_zero_and_one_half_size" src="https://user-images.githubusercontent.com/3244249/152354583-d7b68902-d402-4098-922b-b1a33b07e3e1.png">                                                             | Greyscale images of the digits "1" and "0" - a 2-class subset from the famous[MNIST dataset](http://yann.lecun.com/exdb/mnist/) for handwritten digit classification.          | <img width="120" alt="BinaryMNIST" src="https://user-images.githubusercontent.com/3244249/150808267-3d27eae0-78f2-45f8-8569-cb2561f2c2e9.png">     | [Binary MNIST dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/MNIST)                       |
| [Simple Geometric (circles and triangles)](https://doi.org/10.5281/zenodo.5012824) <img width="20" alt="Simple Geometric Logo" src="https://user-images.githubusercontent.com/3244249/150808842-d35d741e-294a-4ede-bbe9-58e859483589.png"> | Images of circles and triangles for 2-class geometric shape classificaiton. The shapes of varying size and orientation and the background have varying uniform gray levels. | <img width="130" alt="SimpleGeometric" src="https://user-images.githubusercontent.com/3244249/150808125-e1576237-47fa-4e51-b01e-180904b7c7f6.png"> | [Simple geometric shapes dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/geometric_shapes) |
| [Simple Scientific (LeafSnap30)](https://zenodo.org/record/5061353/) <img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/150815639-2da560d4-8b26-4eeb-9ab4-dabf221a264a.png">                      | Color images of tree leaves - a 30-class post-processed subset from the LeafSnap dataset for automatic identification of North American tree species.                       | <img width="600" alt="LeafSnap" src="https://user-images.githubusercontent.com/3244249/150804246-f714e517-641d-48b2-af26-2f04166870d6.png">        | [LeafSnap30 dataset generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/dataset_preparation/LeafSnap/)                     |

</details>

<details><summary>Text</summary>
<p>

| Dataset                                                                                                                                                                                                                            | Description                                                                   | Examples                                                         | Generation                                                          |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- | :--------------------------------------------------------------- | :------------------------------------------------------------------ |
| [Stanford sentiment treebank](https://nlp.stanford.edu/sentiment/index.html) <img width="20" alt="nlp-logo_half_size" src="https://user-images.githubusercontent.com/3244249/152355020-908c04f3-aa99-489d-b87a-7e6b1f586118.png"> | Dataset for predicting the sentiment, positive or negative, of movie reviews. | _This movie was actually neither that funny, nor super witty._ | [Sentiment treebank](https://nlp.stanford.edu/sentiment/treebank.html) |
| [EU-law statements](https://zenodo.org/records/8200000)  <img width="25" alt="nlp-logo_half_size" src="https://avatars.githubusercontent.com/u/133206807?s=48&v=4"> | Reproducibility data for a quantitative study on EU legislation. | _A Member State wishing to grant exemptions referred to in paragraph 6 shall notify the Council in writing_ | [EU legislation strictness analysis](https://github.com/nature-of-eu-rules/eu-legislation-strictness-analysis) |
</details>

<details><summary>Time series</summary>
<p>

| Dataset                                                                                                                                                                                                                | Description                                                                                                                                                    | Examples                                                                                                                                 | Generation                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| [Coffee dataset](https://www.timeseriesclassification.com/description.php?Dataset=Coffee)  <img width="25" alt="Coffe Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/9ab50a0f-5da3-41d2-80e9-70d2c8769162"> | Food spectographs time series dataset for a two class problem to distinguish between Robusta and Arabica coffee beans.                                         | <img width="500" alt="example image" src="https://github.com/dianna-ai/dianna/assets/3244249/763002c5-40ad-48cc-9de0-ea43d7fa8a75"> | [data source](https://github.com/QIBChemometrics/Benchtop-NMR-Coffee-Survey) |
| [Weather dataset](https://zenodo.org/record/7525955) <img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f">                                | The light version of the weather prediciton dataset, which contains daily observations (89 features) for 11 European locations through the years 2000 to 2010. | <img width="500" alt="example image" src="https://github.com/dianna-ai/dianna/assets/3244249/b0a505ac-8a6c-4e1c-b6ad-35e31e52f46d"> | [data source](https://github.com/florian-huber/weather_prediction_dataset)   |

</details>

<details><summary>Tabular</summary>
<p>

| Dataset                                                                                                                                                                                                                | Description                                                                                                                                                    | Examples                                                                                                                                 | Generation                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| [Pengiun dataset](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris) <img width="75" alt="Penguins Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/c7716ad3-f992-4557-80d9-1d8178c7ed57"> | Palmer Archipelago (Antarctica) penguin dataset is a great intro dataset for data exploration & visualization similar to the famous Iris dataset.                                         | <img width="500" alt="example image" src="https://github.com/allisonhorst/palmerpenguins/blob/main/man/figures/README-mass-flipper-1.png"> | [data source](https://github.com/allisonhorst/palmerpenguins) |
| [Weather dataset](https://zenodo.org/record/7525955) <img width="25" alt="Weather Logo" src="https://github.com/dianna-ai/dianna/assets/3244249/3ff3d639-ed2f-4a38-b7ac-957c984bce9f">                                | The light version of the weather prediciton dataset, which contains daily observations (89 features) for 11 European locations through the years 2000 to 2010. | <img width="500" alt="example image" src="https://github.com/dianna-ai/dianna/assets/3244249/b0a505ac-8a6c-4e1c-b6ad-35e31e52f46d"> | [data source](https://github.com/florian-huber/weather_prediction_dataset)   |
| [Land atmosphere dataset](https://zenodo.org/records/12623257) <img width="25" alt="Atmosphere Logo" src="https://github.com/user-attachments/assets/bee353dd-c19a-4aec-a778-4ca3574765f0"> | It contains land-atmosphere variables and latent heat flux (LEtot) simulated by STEMMUS-SCOPE (soil-plant model), version 1.5.0,  over 19 Fluxnet sites and for the year 2014 with hourly intervals. | <img width="500" alt="example image" src="https://github.com/user-attachments/assets/a6e10b08-08d8-4e57-887a-cd4fca9f2ff0"> | [data source](https://zenodo.org/records/12623257)   |
</details>

## Models

**We work with ONNX!** ONNX is a great unified neural network standard which can be used to boost reproducible science. Using ONNX for your model also gives you a boost in performance! In case your models are still in another popular DNN (deep neural network) format, here are some simple recipes to convert them:

* [pytorch and pytorch-lightning](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/pytorch2onnx.ipynb) - use the built-in [`torch.onnx.export`](https://pytorch.org/docs/stable/onnx.html) function to convert pytorch models to onnx, or call the built-in [`to_onnx`](https://pytorch-lightning.readthedocs.io/en/latest/deploy/production_advanced.html) function on your [`LightningModule`](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule) to export pytorch-lightning models to onnx.
* [tensorflow](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/tensorflow2onnx.ipynb) - use the [`tf2onnx`](https://github.com/onnx/tensorflow-onnx) package to convert tensorflow models to onnx.
* [keras](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/keras2onnx.ipynb) - same as the conversion from tensorflow to onnx, the [`tf2onnx`](https://github.com/onnx/tensorflow-onnx) package also supports keras.
* [scikit-learn](https://github.com/dianna-ai/dianna/blob/main/tutorials/conversion_onnx/skl2onnx.ipynb) - use the [`skl2onnx`](https://github.com/onnx/sklearn-onnx) package to scikit-learn models to onnx.

More converters with examples and tutorials can be found on the [ONNX tutorial page](https://github.com/onnx/tutorials).

And here are links to notebooks showing how we created our models on the benchmark datasets:

<details><summary>Images</summary>

| Models                                                    | Generation                                                                                                                                                             |
| :-------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Binary MNIST model](https://zenodo.org/record/5907177)      | [Binary MNIST model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/MNIST/generate_model_binary.ipynb)                |
| [Simple Geometric model](https://zenodo.org/deposit/5907059) | [Simple geometric shapes model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/geometric_shapes/generate_model.ipynb) |
| [Simple Scientific model](https://zenodo.org/record/5907196) | [LeafSnap30 model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/LeafSnap/generate_model.ipynb)                      |
</details>

<details><summary>Text</summary>

| Models                                                               | Generation                                                                                                                                                                 |
|:---------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Movie reviews model](https://zenodo.org/record/5910598)             | [Stanford sentiment treebank model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/movie_reviews/generate_model.ipynb) |
| [Regalatory statement classifier](https://zenodo.org/record/8200001) | [EU-law regulatory-statement-classification](https://github.com/nature-of-eu-rules/regulatory-statement-classification)                                                    |

</details>

<details><summary>Time series</summary>

| Models                                                    | Generation                                                                                                                                                        |
| :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Coffee model](https://zenodo.org/records/10579458)                          | [Coffee model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/coffee/generate_model.ipynb)                       |
| [Season prediction model](https://zenodo.org/record/7543883)                 | [Season prediction model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/season_prediction/generate_model.ipynb) |
| [Fast Radio Burst classification model](https://zenodo.org/records/10656614) | [Fast Radio Burst classification model generation](https://doi.org/10.3847/1538-3881/aae649) |

</details>

<details><summary>Tabular</summary>

| Models                                                    | Generation                                                                                                                                                        |
| :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Penguin model    (classification)](https://zenodo.org/records/10580743)                         | [Penguin model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/penguin_species/generate_model.ipynb)          |
| [Sunshine hours prediction model (regression)](https://zenodo.org/records/10580833) | [Sunshine hours prediction model generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/model_generation/sunshine_prediction/generate_model.ipynb) |
| [Latent heat flux prediction model (regression)](https://zenodo.org/records/12623257) | [Latent heat flux prediction model](doi:10.5281/zenodo.12623256/stemmus_scope_emulator_model_LEtot.onnx)   |

</details>

**_We envision the birth of the ONNX Scientific models zoo soon..._**

## Tutorials

DIANNA supports different data modalities and XAI methods (explainers). We have evaluated many explainers using objective criteria (see the [How to find your AI explainer](https://blog.esciencecenter.nl/how-to-find-your-artificial-intelligence-explainer-dbb1ac608009) blog-post). The table below contains links to the relevant XAI method's papers (for some explanatory videos on the methods, please see [tutorials](./tutorials)). The DIANNA [tutorials](./tutorials) cover each supported method and data modality on a least one dataset using the default or tuned [hyperparameters](./tutorials/README.md#important-hyperparameters). Our plans to expand DIANNA with more data modalities and explainers are given in the [ROADMAP](https://dianna.readthedocs.io/en/latest/ROADMAP.html).

<!-- see issue: https://github.com/dianna-ai/dianna/issues/142, also related issue: https://github.com/dianna-ai/dianna/issues/148 -->

| Data \ XAI | [RISE](http://bmvc2018.org/contents/papers/1064.pdf) | [LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) | [KernelSHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) |
| :--------- | :------------------------------------------------ | :----------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| Images     | ✅                                                | ✅                                                                 | ✅                                                                                                 |
| Text       | ✅                                                | ✅                                                                 |                                                                                                     |
| Timeseries | ✅                                                | ✅                                                                 |                                                                                                     |
| Tabular    | planned                                           | ✅                                                                 | ✅                                                                                                  |
| Embedding  | *inspired by RISE in [distance_explainer](https://github.com/dianna-ai/distance_explainer)  |                                                                     |                                                                                                     |
| Graphs*    | next steps                                        |    ...                                                              |     ...                                                                                             |

[LRP](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable) and [PatternAttribution](https://arxiv.org/pdf/1705.05598.pdf) also feature in the top 5 of our thoroughly evaluated explainers.
Also [GradCAM](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)) has been recently found to be *semantically continous*! **Contributing by adding these and more (new) post-hoc explainability methods on ONNX models is very welcome!**


### Scientific use-cases
Our goal is that the scientific community embrases XAI as a source for novel and unexplored perspectives on scientific problems.
Here, we offer [tutorials](./tutorials) on specific scientific use-cases of uisng XAI:

| Use-case (data) \ XAI                                              | [RISE](http://bmvc2018.org/contents/papers/1064.pdf) | [LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) | [KernelSHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) |
|:-------------------------------------------------------------------|:-----------------------------------------------------| :---------------------------------------------------------------------| :-------------------------------------------------------------------------------------------------------|
| Biology (Phytomorphology): Tree Leaves classification (images)     |                                                      |            ✅                                                        |                                                                                                         |
| Astronomy: Fast Radio Burst detection (timeseries)                 | ✅                                                    |                                                                       |                                                                                                         |
| Land-atmosphere modeling: Latent heat flux prediction (tabular)    |                                                      |                                                                    | ✅                                                                                                     |
| Social sciences: EU-law regulatory statement classification (text) |                                                      |  ✅                                                                  |                                                                                                     |
| Climate                                                            | planned                                              |   ...                                                                 |       ...                                                                                              |

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
