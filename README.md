[![build](https://github.com/dianna-ai/dianna/actions/workflows/build.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/build.yml)
[![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=coverage)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna)
[more badges](badges.md)

# DIANNA: Deep Insight And Neural Network Analysis

Modern scientific challenges are often tackled with (Deep) Neural Networks (DNN). Despite their high predictive accuracy, DNNs lack inherent explainability. Many DNN users, especially scientists, do not harvest DNNs power because of lack of trust and understanding of their working. 

Meanwhile, the eXplainable AI (XAI) methods offer some post-hoc interpretability and insight into the DNN reasoning. This is done by quantifying the relevance of individual features (image pixels, words in text, etc.) with respect to the prediction. These "relevance heatmaps" indicate how the network has reached its decision directly in the input modality (images, text, speech etc.) of the data. 

There are many Open Source Software (OSS) implementations of these methods, alas, supporting a single DNN format and the libraries are known mostly by the AI experts. The DIANNA library supports the best XAI methods in the context of scientific usage providing their OSS implementation based on the ONNX standard and demonstrations on benchmark datasets. Representing visually the captured knowledge by the AI system can become a source of (scientific) insights. 

## How to use dianna

The project setup is documented in [project_setup.md](https://github.com/dianna-ai/dianna/blob/main/project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

## Installation [![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/)

To install dianna directly from the GitHub repository, do:

```console
python3 -m pip install git+https://github.com/dianna-ai/dianna.git
```

For development purposes, when you first clone the repository locally, it may be more convenient to install in editable mode using pip's `-e` flag:

```console
git clone https://github.com/dianna-ai/dianna.git
cd dianna
python3 -m pip install -e .
```

## Documentation [![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest)


## Contributing

If you want to contribute to the development of dianna,
have a look at the [contribution guidelines](https://github.com/dianna-ai/dianna/blob/main/CONTRIBUTING.md).

## How to cite us [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592606.svg)](https://doi.org/10.5281/zenodo.5592606)


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
