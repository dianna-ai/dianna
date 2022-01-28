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
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=coverage)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna)
[![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/)
[![supported python versions](https://img.shields.io/pypi/pyversions/dianna)](https://pypi.python.org/project/dianna/)


[![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5542/badge)](https://bestpractices.coreinfrastructure.org/projects/5542)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592607.svg)](https://zenodo.org/record/5592607)
[![more badges badge](https://img.shields.io/badge/more-badges-lightgrey)](badges.md)

# DIANNA: Deep Insight And Neural Network Analysis
<!-- TODO: add main points and then expand, see issue https://github.com/dianna-ai/dianna/issues/137 -->

## Why DIANNA? 
<!-- TO DO: edit the proposal text into something clear and simpler -->

DIANNA software is addressing needs of both (X)AI reseachers and mostly the various domains scientists who are using or will use AI models for their research without being experts in (X)AI. There are numerous python OSS [XAI libraries](https://github.com/wangyongjie-ntu/Awesome-explainable-AI#python-librariessort-in-alphabeta-order), the Awsome explainable AI features 57 of them, of which 9 have more than 2000 github stars:
[SHAP](https://github.com/slundberg/shap)![](https://img.shields.io/github/stars/slundberg/shap.svg?style=social),
[LIME](https://github.com/marcotcr/lime) ![](https://img.shields.io/github/stars/marcotcr/lime.svg?style=social),
[pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)![](https://img.shields.io/github/stars/utkuozbulak/pytorch-cnn-visualizations?style=social),
[InterpretML](https://github.com/interpretml/interpret) ![](https://img.shields.io/github/stars/InterpretML/interpret.svg?style=social)
[Lucid](https://github.com/tensorflow/lucid) ![](https://img.shields.io/github/stars/tensorflow/lucid.svg?style=social),
[Pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)![](https://img.shields.io/github/stars/jacobgil/pytorch-grad-cam?style=social),
[Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox) ![](https://img.shields.io/github/stars/yosinski/deep-visualization-toolbox?style=social),
[Captum](https://github.com/pytorch/captum) ![](https://img.shields.io/github/stars/pytorch/captum.svg?style=social), 
[ELI5](https://github.com/TeamHG-Memex/eli5) ![](https://img.shields.io/github/stars/TeamHG-Memex/eli5.svg?style=social),etc..
These libraries currently have serious limitaitons in respect to the usage by the varoius scientific communities:

* A single XAI method is implemented
While SHAP, LIME, Pytorch-grad-cam, DeepLIFT, etc. enjoy enourmous popularity, the spesific methods are not always suitable for the research task and/or data modality (?). Most importantly, different methods approach the AI explaianbility differently and so a single method provides limited explaianbility capacity. Which is the best method to use?
* A single DNN format/framework is supported
Lucid supports only [Tensoflow](https://www.tensorflow.org/), Captum is the library for [PyTorch](https://pytorch.org/) users, [iNNvstigate](https://github.com/albermax/innvestigate) has been used by Keras users. Pytorch-grad-cam supports a single method for a single format and Pytorch-cnn-visualizaiton even limits the choice to a single DNN type- CNN. While this is not an issue for the current most popular framework communities, not all are supported by mature libraries (e.g. ) and most importantly are not "future-proof", e.g. Caffe has been the most polular framework by the computer vision comminuty in , while now is the era of Tensorflow, Pytorch and Keras.  
* The choice of supported XAI methods seems random
ELI5, for example, supports multiple frameworks/formats and XAI methods, but is not clear how the selection of these methods has been made. Futhermore the library is not maintained since 2020 and hence lacks any methods proposed since. Again, which are the best methods to use?
* Requires (X)AI expertise
For example the DeepVis Toolbox requires DNN knowledge and is used only by AI experts mostly within the computer vision community (which is large and can explain the number of stars). 

All of the above issues have a common reason: the libraries have been developed by (X)AI reseachers for the (X)AI reseachrs.
 
Even for the purpose of advancing (X)AI research, systematic study of the properties of the "heatmaps" is lacking and currently in the XAI publicaitons often the human interpretation is intertwined with the one of the exlainer. Suitable datasets for such a study are lacking: the popular and simplest dataset used in publicaitons,  MNIST, is too complex for such a prurpose with 10 classes and no structural content variation. The XAI community does not publish work on simple scientific datasets, which would make the technology understanable and trustable by non-expert scientists on intuitive level.


Solutions:
1.	To demonstrate the usefulness and properties of the heatmaps on an intuitive level we propose: simple geometrical and simple scientific– subset of LeafSnap datasets. Tree species classification on the LeafSnap data is a good example of problem tackled with both classical Computer Vision and the superior DL method.
2.	Recently, several systematically defined criteria for evaluation of the XAI approaches have been proposed with LIME analyzed as example. Analysis of the state-of-the-art XAI methods will highlight the best.
3.	DIANNA is a library conforming with the ONNX standard. There are many ONNX tools available as OSS including the ONNX model zoo and ONNX converters from Keras and TensorFlow. PyTorch also offers built-in PyTorch to ONNX export.

## Installation 

To install DIANNA directly from the GitHub repository, do:

```console
python3 -m pip install git+https://github.com/dianna-ai/dianna.git
```

For development purposes, when you first clone the repository locally, it may be more convenient to install in editable mode using pip's `-e` flag:

```console
git clone https://github.com/dianna-ai/dianna.git
cd dianna
python3 -m pip install -e .
```


## Datasets
DIANNA comes with simple datasets. Their main goal is to provide intuitive insight into the working of the XAI methods. They can be used as benchmarks for evaluation and comparison of existing and new XAI methods.

### Images
|Dataset|Description|Examples|Generation|
|:-----|:----|:---|:----|
|Binary MNIST | Greyscale images of the digits "1" and "0" - a 2-class subset from the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for handwritten digit classification. |<img width="120" alt="BinaryMNIST" src="https://user-images.githubusercontent.com/3244249/150808267-3d27eae0-78f2-45f8-8569-cb2561f2c2e9.png">| [Binary MNIST dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/MNIST)|
|[Simple Geometric (circles and triangles)](https://doi.org/10.5281/zenodo.5012824) <img width="20" alt="Simple Geometric Logo" src="https://user-images.githubusercontent.com/3244249/150808842-d35d741e-294a-4ede-bbe9-58e859483589.png"> | Images of circles and triangles for 2-class geometric shape classificaiton. The shapes of varying size and orientation and the background have varying uniform gray levels.  | <img width="130" alt="SimpleGeometric" src="https://user-images.githubusercontent.com/3244249/150808125-e1576237-47fa-4e51-b01e-180904b7c7f6.png">| [Simple geometric shapes dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/geometric_shapes) | 
|[Simple Scientific (LeafSnap30)](https://zenodo.org/record/5061353/)<img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/150815639-2da560d4-8b26-4eeb-9ab4-dabf221a264a.png"> | Color images of tree leaves - a 30-class post-processed subset from the LeafSnap dataset for automatic identification of North American tree species.|<img width="600" alt="LeafSnap" src="https://user-images.githubusercontent.com/3244249/150804246-f714e517-641d-48b2-af26-2f04166870d6.png">| [LeafSnap30 dataset generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/dataset_preparation/LeafSnap/)|

### Text
|Dataset|Description|Examples|Generation|
|:-----|:----|:---|:----|
| [Stanford sentiment treebank](https://nlp.stanford.edu/sentiment/index.html)|Dataset for predicting the sentiment, positive or negative, of movie reviews. | _This movie was actually neither that funny, nor super witty._|[Sentiment treebank](https://nlp.stanford.edu/sentiment/treebank.html)|

## ONNX models
<!-- TODO: Add all links, see issue https://github.com/dianna-ai/dianna/issues/135 -->

**We work with ONNX!** ONNX is a great unified neural network standard which can be used to boost reproducible science. Using ONXX for your model also gives you a boost in performance! In case your models are still in another popular DNN (deep neural network) format, here are some simple recipes to convert them:
* pytorch
* tensorflow
* keras
* scikit-learn

And here are links to notebooks showing how we created our models on the benchmark datasets:

### Images
* Binary MNIST model
* Simple Geometric model
* Simple Scientific model

### Text
* Movie reviews model

**_We envision the birth of the ONNX Scientific models zoo soon..._**

## Tutorials
DIANNA supports different data modalities and XAI methods. The table contains links to the relevant XAI method's papers. There are DIANNA [tutorials](./tutorials) covering each supported method and data modality on a least one dataset. Our future plans to expand DIANNA with more data modalities and XAI methods are given at the [ROADMAP.md](./ROADMAP.md).

<!-- see issue: https://github.com/dianna-ai/dianna/issues/142, also related issue: https://github.com/dianna-ai/dianna/issues/148 -->

|Data \ XAI|[RISE](http://bmvc2018.org/contents/papers/1064.pdf)|[LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)|[KernelSHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)|
|:-----|:---|:---|:---|
|Images|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Text|:white_check_mark:|:white_check_mark:|planned|
|Embedding|coming soon|coming soon|coming soon|
|Timeseries|planned|planned|planned|
|Tabular|planned|planned|planned|

[LRP](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable) and [PatternAttribution](https://arxiv.org/pdf/1705.05598.pdf) also feature in the top 5 of our thoroughly evaluated XAI methods using objective critera (details in coming blog-post). **Contributing by adding these and more (new) post-hoc explainability methods on ONNX models is very welcome!**

## Reference documentation 

For detailed information on using specific DIANNA functions, please visit the [Sphinx documentation page hosted at Readthedocs](https://dianna.readthedocs.io/en/latest).

## Contributing

If you want to contribute to the development of DIANNA,
have a look at the [contribution guidelines](https://github.com/dianna-ai/dianna/blob/main/CONTRIBUTING.md).

## How to cite us 

If you use this package for your scientific work, please consider citing it as:

    Ranguelova, Elena, Bos, Patrick, Liu, Yang, Meijer, Christiaan, & Oostrum, Leon. (2021). dianna (*[VERSION YOU USED]*). Zenodo. https://zenodo.org/record/5592607

See also the [Zenodo page](https://zenodo.org/record/5592607) for exporting the citation to BibTteX and other formats.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
