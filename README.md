[![build](https://github.com/dianna-ai/dianna/actions/workflows/build.yml/badge.svg)](https://github.com/dianna-ai/dianna/actions/workflows/build.yml)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=dianna-ai_dianna&metric=coverage)](https://sonarcloud.io/dashboard?id=dianna-ai_dianna)
[![workflow pypi badge](https://img.shields.io/pypi/v/dianna.svg?colorB=blue)](https://pypi.python.org/project/dianna/)
[![supported python versions](https://img.shields.io/pypi/pyversions/dianna)](https://pypi.python.org/project/dianna/)


[![Documentation Status](https://readthedocs.org/projects/dianna/badge/?version=latest)](https://dianna.readthedocs.io/en/latest/?badge=latest)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5542/badge)](https://bestpractices.coreinfrastructure.org/projects/5542)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592607.svg)](https://zenodo.org/record/5592607)
[![more badges badge](https://img.shields.io/badge/more-badges-lightgrey)](badges.md)

# DIANNA: Deep Insight And Neural Network Analysis
(TO DO: main points and then expand)

## Why DIANNA? 
(TO DO: edit the proposal text into something clear and simpler)
Issues:
1.	The properties of the heatmaps are not studied and the human interpretation is intertwined with the XAI’s. Suitable datasets are lacking: the popular MNIST benchmark is too complex for the task (10 classes and no structural content variation). The XAI literature does not consider simple scientific “benchmarks”.
2.	Which is the “best” explainability method? There is no agreement in the XAI community [23]. The libraries offer different subset of XAI methods not chosen systematically.
3.	The available OSS (not for all methods) implementations support a single DNN format/framework, e.g. iNNvestigate supports only Keras, while Captum supports PyTorch. 
4.	Not many demonstrators of XAI exist, except from LRP and RISE

Solutions:
1.	To demonstrate the usefulness and properties of the heatmaps on an intuitive level we propose: simple geometrical, e.g. Triangles and Squares Rotation and Scaling (Figure 4, Left) [I] and simple scientific– subset of LeafSnap (Figure 4, Right) datasets. Tree species classification on the LeafSnap data is a good example of problem tackled with both classical Computer Vision [24] and the superior DL method [25].
2.	Recently, several systematically defined criteria for evaluation of the XAI approaches have been proposed with LIME analyzed as example [12]. Analysis of the state-of-the-art XAI methods will highlight the best.
3.	DIANNA will be a library conforming with the ONNX standard [11]. There are many ONNX tools available as OSS including the ONNX model zoo and ONNX converters from Keras and TensorFlow. PyTorch also offers built-in PyTorch to ONNX export.
4.	A web demonstrator will be created in a next phase of the project (due to current budget cuts). 

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
DIANNA comes with simple datasets. Their main goal is to provide intuitive insight to the working of the XAI methods. They can be used as benchmakrs for evaluation and comparision of exsisting and new XAI methods:

### Images
|Dataset|Description|Examples|Generation|
|:-----|:----|:---|:----|
|Binary MNIST <img width="20" alt="BinaryMNIST Logo" src="https://user-images.githubusercontent.com/3244249/150810986-8e94f1f7-0647-4ce4-ab0e-474b092480a5.png">| Greyscale images of the digits "1" and "0" - a 2 class subset from the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for handwritten digit classification. |<img width="120" alt="BinaryMNIST" src="https://user-images.githubusercontent.com/3244249/150808267-3d27eae0-78f2-45f8-8569-cb2561f2c2e9.png">| [Binary MNIST dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/MNIST)|
|[Simple Geometric (circles and triangles)](https://doi.org/10.5281/zenodo.5012824) <img width="20" alt="Simple Geometric Logo" src="https://user-images.githubusercontent.com/3244249/150808842-d35d741e-294a-4ede-bbe9-58e859483589.png"> | Images of circules and triangles for 2 class geometric shape classificaiton. The shapes of varying size and orientation and the background have varying uniform gray levels.  | <img width="130" alt="SimpleGeometric" src="https://user-images.githubusercontent.com/3244249/150808125-e1576237-47fa-4e51-b01e-180904b7c7f6.png">| [Simple geometric shapes dataset generation](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/geometric_shapes) | 
|[Simple Scientific (LeafSnap30)](10.5281/zenodo.5061352)<img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/150815639-2da560d4-8b26-4eeb-9ab4-dabf221a264a.png"> | Color images of tree leaves - a 30 class post-processed subset from the LeafSnap dataset for automatic identification of North American tree species.|<img width="600" alt="LeafSnap" src="https://user-images.githubusercontent.com/3244249/150804246-f714e517-641d-48b2-af26-2f04166870d6.png">| [LeafSnap30 dataset generation](https://github.com/dianna-ai/dianna-exploration/blob/main/example_data/dataset_preparation/LeafSnap/)|

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
