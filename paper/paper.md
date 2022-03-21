---
title: 'DIANNA: Deep Insight And Neural Network Analysis'
tags:
  - Python
  - explainable AI
  - Deep Neural Networks
  - ONNX
  - benchmark datasets
authors:
  - name: Elena Ranguelova 
    orcid: 0000-0002-9834-1756
    affiliation: 1
  - name: Christiaan Meijer
    orcid: 0000-0002-5529-5761
    affiliation: 1
  - name: Leon Oostrum
    orcid: 0000-0001-8724-8372
    affiliation: 1
  - name: Yang Liu
    orcid: 0000-0002-1966-8460
    affiliation: 1
  - name: Patrick Bos 
    orcid: 0000-0002-6033-960X
    affiliation: 1
  - name: Giulia Crocioni 
    orcid: 0000-0002-0823-0121
    affiliation: 1
  - name: Matthieu Laneuville 
    orcid: 0000-0001-6022-0046
    affiliation: 2
  - name: Bryan Cardenas Guevara
    orcid: 0000-0001-9793-910X
    affiliation: 2
  - name: Rena Bakhshi 
    orcid: 0000-0002-2932-3028 
    affiliation: 1   
  - name: Damian Podareanu
    orcid: 0000-0002-4207-8725
    affiliation: 2
affiliations:
 - name: Netherlands eScience Center, Amsterdam, the Netherlands
   index: 1
 - name: SURF, Amsterdam, the Netherlands
   index: 2
date: 18 March 2022 # put date here
bibliography: paper.bib
---

# Summary

The growing demand from science and industry inspired rapid advances in Artificial Intelligence (AI). The increased use of AI and the reliability and trust required by automated decision making and standards of scientific rigour, led to the boom of eXplainable Artificial Intelligence (XAI). [DIANNA (Deep Insight And Neural Network Analysis)](https://dianna.readthedocs.io/en/latest/) is the first Python package of systematically selected XAI methods supporting the [Open Neural Networks Exchange (ONNX)](https://onnx.ai/) format. DIANNA is built by and designed for all research software engineers and researchers, also non-AI experts.

# Statement of need

AI systems have been increasingly used in a wide variety of fields, including such data-sensitive areas as healthcare [@alshehrifatima], renewable energy [@kuzlumurat], supply chain [@toorajipourreza] and finance. Automated decision-making and scientific research standards require reliability and trust of the AI technology [@xuwei]. Especially in AI-enhanced research, a scientist need to be able to trust a high-performant, but opaque AI model used for automation of their data processing pipeline. In addition, XAI has the potential for helping any scientist to "find new scientific discoveries in the analysis of their data” [@hey]. Furthermore, tools for supporting repeatable science are of high demand  [@Feger2020InteractiveTF].

DIANNA addresses these needs of researchers in various scientific domains who make use of AI models, especially supporting non-AI experts. DIANNA provides a Python-based, user-friendly, and uniform interface to several XAI methods. To the best of our knowledge, it is the only library using Open Neural Network Exchange (ONNX) [@onnx], the open-source, framework-agnostic standard for AI models, which supports repeatability of scientific research.

# State of the field

There are numerous Python XAI libraries, many are listed in the Awesome explainable AI [@awesomeai] repository. Popular and widely used packages are Pytorch [@pytorch], LIME [@ribeirolime], Captum [@kokhlikyan2020captum], Lucid [@tflucid], SHAP [@lundbergshap], InterpretML [@nori2019interpretml], PyTorch CNN visualizaitons [@uozbulak_pytorch_vis_2021] Pytorch GradCAM [@jacobgilpytorchcam], Deep Visualization Toolbox [@yosinski-2015-ICML-DL-understanding-neural-networks], ELI5 [@eli5]. However, these libraries have limitations that complicate adoption by scientific communities:

- **Single XAI method** or **single data modality.**
While libraries such as SHAP, LIME, Pytorch GradCAM. have gained great popularity, their methods are not always suitable for the research task and/or data modality. For example, GradCAM is applicable only to images. Most importantly, each library in that class addresses AI explainability with a different method, complicating comparison between methods. 
- **Single Deep Neural Network (DNN) format/framework/architecture.** Many XAI libriaries support a single DNN format: Lucid supports only TensorFlow [@tf], Captum - PyTorch [@pytorch] and iNNvestigate [@innvestigatenn] is aimed at Keras users exclusively. Pytorch GradCAM supports a single method for a single format and Convolutional Neural Network Visualizations even limits the choice to a single DNN type. While this is not an issue for the current most popular framework communities, not all mature libraries support a spectrum of XAI methods. Most importantly, tools that support a single framework are not "future-proof". For instance, Caffe [@jia2014caffe] was the most popular framework in the computer vision (CV) community in 2018, but it has since been abandoned.
- **Unclear choice of supported XAI methods.** ELI5 supports multiple frameworks/formats and XAI methods, but it is unclear how the selection of these methods was made. Furthermore, the library has not been maintained since 2020, so any methods in the rapidly changing XAI field proposed since then are missing.
- **AI expertise is necessary.** The Deep Visualization Toolbox requires DNN knowledge and is only used by AI experts mostly within the CV community.


In addition, on more fundamental level, the results of XAI research does not help to make the technology understandable and trustworthy for non (X)AI experts:

- **Properties of the output of the explainer.** There is no commonly accepted methodology to systematically study XAI methods and their output.
- **Human interpretation intertwined with the one of the explainer.** This is a major problem in the current XAI literature, and there has been limited research to define what constitutes a meaningful explanation in the context of AI systems [@joylu].
- **Lack of suitable (scientific) datasets.** The most popular and simplest dataset used as "toy-example" is the MNIST dataset of handwritten digits [@mnistdataset], composed of 10 classes and with no structural variation in the content. Such a dataset is too complex for non-AI experts to intuitively understand the XAI output and simultaneously too far from scientific research data.
- **Plethora of current AI model formats.** The amount and the speed with which they become obsolete is another important show-stopper for reproducibility of AI-enabled science.
- **Lack of funding.** Some libraries, such as iNNvestigate, are becoming obsolete due to the lack of support for research projects that sponsored their creation.


# Key Features

![High level architecture of DIANNA](https://user-images.githubusercontent.com/3244249/158770366-a624d1e0-2eae-43cc-aeb5-bfa33b50b3e4.png)

DIANNA is an open source XAI Python package with the following key characteristics:

- **Systematically chosen diverse set of XAI methods.**  We have used a relevant subset of the thorough objective and systematic evaluation criteria defined in [@peterflatch]. Several complementary and model-architecture agnostic state-of-the-art XAI methods have been chosen and included in DIANNA [@ranguelova_how_2022].
- **Multiple data modalities.** DIANNA supports images and text, we will extend the input data modalities to embeddings, time-series, tabular data and graphs. This is particularly important to scientific researchers, whose data are in domains different than the  classical examples from CV and natural language processing communities.
- **Open Neural Network Exchange (ONNX) format.** ONNX is the de-facto standard format for neural network models. Not only is the use of ONNX very beneficial for interoperability, enabling reproducible science, but it is also compatible with runtimes and libraries designed to maximize performance across hardware. To the best of our knowledge, DIANNA is the first and only XAI library supporting ONNX.
- **Simple, intuitive benchmark datasets.** We have proposed two new datasets which enable systematic research of the properties of the XAI methods' output and understanding on an intuitive level: Simple Geometric Shapes [@oostrum_leon_2021_5012825] and LeafSnap30 [@ranguelova_elena_2021_5061353]. The classification of tree species on LeafSnap data is a great example of a simple scientific problem tackled with both classical CV and a deep learning method, where the latter outperforms, but needs explanations.  DIANNA also uses well-established benchmarks: a simplified MNIST with 2 distinctive classes only and the Stanford Sentiment Treebank [@socher-etal-2013-recursive].
- **User-friendly interface.** DIANNA wraps all XAI methods with a common API.
- **Modular architecture, extensive testing and compliance with modern software engineering practices.** It is very easy for new XAI methods which do not need to access the ONNX model internals to be added to DIANNA. For relevance-propagation type of methods, more work is needed within the ONNX standart [@levitan_onnx_2020] and we hope our work will boost the development growth of ONNX (scientific) models. We welcome the XAI research community to contribute to these developments via DIANNA.
- **Thorough documentation.** The package includes user and developer documentation. It also provides instructions for conversion between ONNX and Tensorflow, Pytorch, Keras or Scikit-learn.

# Used by

DIANNA is currently used in the "Recognizing symbolism in Turkish television drama" project [@turkishdrama]. An important task is the development of an effective model for detecting and recognizing symbols in videos. DIANNA is used to increase insight into the AI models in order to explore how to improve them.

DIANNA is also currently used in the "Visually grounded models of spoken language" project, which builds on earlier work from [@chrupala+17-representations;@alishahi+17;@chrupala18;@chrupala+19].
The goal is a multi-modal model by projecting image and sound data into a common embedded space. Within DIANNA, we are developing XAI methods to visualize and explain these embedded spaces in their complex multi-modal network contexts.

Finally, DIANNA was also used in the EU-funded [Examode](https://www.examode.eu/) medical research project [@bryancardenas]. It  deals with very large data sets and since it aims to support physicians in their decision-making, it needs transparent and trustworthy models.

# Acknowledgements

This work was supported by the [Netherlands eScience Center](https://www.esciencecenter.nl/) and [SURF](https://www.surf.nl/en).

# References

[//]: # "All the refs need to be put in paper.bib file ([open PR #241](https://github.com/dianna-ai/dianna/pull/241)) and cited above using this notation: [@bibentry]."
