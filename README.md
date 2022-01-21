[![build](https://github.com/dianna-ai/dianna/actions/workflows/build.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/build.yml)
[![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=coverage)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna)

[![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5542/badge)](https://bestpractices.coreinfrastructure.org/projects/5542)
[![more badges badge](https://img.shields.io/badge/more-badges-lightgrey)](badges.md)

# DIANNA: Deep Insight And Neural Network Analysis

Modern scientific challenges are often tackled with (Deep) Neural Networks (DNN). Despite their high predictive accuracy, DNNs lack inherent explainability. Many DNN users, especially scientists, do not harvest DNNs power because of lack of trust and understanding of their working. 

Meanwhile, the eXplainable AI (XAI) methods offer some post-hoc interpretability and insight into the DNN reasoning. This is done by quantifying the relevance of individual features (image pixels, words in text, etc.) with respect to the prediction. These "relevance heatmaps" indicate how the network has reached its decision directly in the input modality (images, text, speech etc.) of the data. 

There are many Open Source Software (OSS) implementations of these methods, alas, supporting a single DNN format and the libraries are known mostly by the AI experts. The DIANNA library supports the best XAI methods in the context of scientific usage providing their OSS implementation based on the ONNX standard and demonstrations on benchmark datasets. Representing visually the captured knowledge by the AI system can become a source of (scientific) insights. 

## How to use DIANNA

The project setup is documented in [project_setup.md](https://github.com/dianna-ai/dianna/blob/main/project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

## Installation [![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/) [![supported python versions](https://img.shields.io/pypi/pyversions/dianna)](https://pypi.python.org/project/dianna/)

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


## Benchmark datasets
(TODO: add a sentence and/or image for each and the Zenodo and/or original links and links to notebooks for creating them)

DIANNA offers to use simple benchmark datasets for evaluating and comparing the XAI methods:

### Images
* Binary (2-class) MNIST
* Simple Geometric (triangles and circles)
* Simple Scientific (LeafSnap30)

### Text
* Movie reviews treebank

## ONNX models
(TODO: Add all links) 

**We work with ONNX!** ONNX is a great unified NN standard which can be used to boost reproducible science. Using ONXX for your model also gives you a boost in performance! In case your models are still in another popular DNN format, here are some simple recipes to convert them:
* pytorch
* tensorflow
* keras
* scikit-learn

And here are how we created our models on the benchmark datasets:

### Images
* Binary MNIST model
* Simple Geometric model
* Simple Scientific model

### Text
* Movie reviews model

**_We envision the birth of the ONNX Scientific models zoo soon..._**

## Tutorials
DIANNA supports the following data modalities and XAI methods (the table contains links to the relevant tutorials or plans):
(TODO: fix links to tutorials in table then ready; check link to roadmap.md after merging)

|Data \ XAI|RISE|LIME|KernelSHAP|
|:-----|:---|:---|:---|
|Images|[:white_check_mark:](https://github.com/dianna-ai/dianna/tree/main/tutorials)|[:white_check_mark:](https://github.com/dianna-ai/dianna/tree/main/tutorials)|[:white_check_mark:](https://github.com/dianna-ai/dianna/tree/main/tutorials)|
|Text|[:white_check_mark:](https://github.com/dianna-ai/dianna/tree/main/tutorials)|[:white_check_mark:](https://github.com/dianna-ai/dianna/tree/main/tutorials)|[planned](https://github.com/dianna-ai/dianna/blob/94-improve-readme/ROADMAP.md)|
|Embedding|[coming soon](https://github.com/dianna-ai/dianna/blob/94-improve-readme/ROADMAP.md)|[coming soon](https://github.com/dianna-ai/dianna/blob/94-improve-readme/ROADMAP.md)|[coming soon](https://github.com/dianna-ai/dianna/blob/94-improve-readme/ROADMAP.md)|
|Timeseries|[planned](https://github.com/dianna-ai/dianna/blob/94-improve-readme/ROADMAP.md)|[planned](https://github.com/dianna-ai/dianna/blob/94-improve-readme/ROADMAP.md)|[planned](https://github.com/dianna-ai/dianna/blob/94-improve-readme/ROADMAP.md)|

LRP and PatternAttribution also feature in the top 5 of our thoroughly evaluated using objective critera XAI methods (details in coming blog-post). Contributing by adding these and more (new) post-hoc explaianbility methods on ONNX models is very welcome!

## Reference documentation [![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest)

For detailed information on using specific functions in DIANNA, please visit the [Sphinx documentation page hosted at Readthedocs](https://dianna.readthedocs.io/en/latest).

## Contributing

If you want to contribute to the development of DIANNA,
have a look at the [contribution guidelines](https://github.com/dianna-ai/dianna/blob/main/CONTRIBUTING.md).

## How to cite us [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592606.svg)](https://doi.org/10.5281/zenodo.5592606)

If you use this package for your scientific work, please consider citing it as:

    Ranguelova, Elena, Bos, Patrick, Liu, Yang, Meijer, Christiaan, & Oostrum, Leon. (2021). dianna (*[VERSION YOU USED]*). Zenodo. https://doi.org/10.5281/zenodo.5592606

See also the [Zenodo page] (https://doi.org/10.5281/zenodo.5592606) for exporting the citation to BibTteX and other formats.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
