# DIANNA: Deep Insight And Neural Network Analysis

Modern scientific challenges are often tackled with (Deep) Neural Networks (DNN). Despite their high predictive accuracy, DNNs lack inherent explainability. Many DNN users, especially scientists, do not harvest DNNs power because of lack of trust and understanding of their working. 

Meanwhile, the eXplainable AI (XAI) methods offer some post-hoc interpretability and insight into the DNN reasoning. This is done by quantifying the relevance of individual features (image pixels, words in text, etc.) with respect to the prediction. These "relevance heatmaps" indicate how the network has reached its decision directly in the input modality (images, text, speech etc.) of the data. 

There are many Open Source Software (OSS) implementations of these methods, alas, supporting a single DNN format and the libraries are known mostly by the AI experts. The DIANNA library supports the best XAI methods in the context of scientific usage providing their OSS implementation based on the ONNX standard and demonstrations on benchmark datasets. Representing visually the captured knowledge by the AI system can become a source of (scientific) insights. 

## How to use dianna

The project setup is documented in [project_setup.md](https://github.com/dianna-ai/dianna/blob/main/project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

## Installation

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

## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/dianna-ai/dianna) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/dianna-ai/dianna)](https://github.com/dianna-ai/dianna) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-dianna-00a3e3.svg)](https://www.research-software.nl/software/dianna) [![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592606.svg)](https://doi.org/10.5281/zenodo.5592606) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=alert_status)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=coverage)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/dianna-ai/dianna/actions/workflows/build.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](https://github.com/dianna-ai/dianna/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](https://github.com/dianna-ai/dianna/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](https://github.com/dianna-ai/dianna/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/markdown-link-check.yml) |

## Documentation

Include a link to your project's full documentation here.

## Contributing

If you want to contribute to the development of dianna,
have a look at the [contribution guidelines](https://github.com/dianna-ai/dianna/blob/main/CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
