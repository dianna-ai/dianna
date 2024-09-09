<!-- this file contains the Statement of Need for JOSS -->

# Why DIANNA?

## Context

There are numerous other python OSS [eXplainable AI (XAI) libraries](https://github.com/wangyongjie-ntu/Awesome-explainable-AI#python-librariessort-in-alphabeta-order), the Awsome explainable AI (XAI) features many of them, the most popular being:
[SHAP](https://github.com/slundberg/shap),
[LIME](https://github.com/marcotcr/lime),
[pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations),
[InterpretML](https://github.com/interpretml/interpret),
[Lucid](https://github.com/tensorflow/lucid),
[Pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam),
[Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox),
[Captum](https://github.com/pytorch/captum),
[ELI5](https://github.com/TeamHG-Memex/eli5), etc.

These libraries currently have serious limitations in respect to the usage by the various *scientific communities*:

* **A single XAI method or single data modality is implemented**

While SHAP, LIME, Pytorch-grad-cam, DeepLIFT, etc. enjoy enormous popularity, the specific methods are not always suitable for the research task and/or data modality. Most importantly, different methods approach the AI explainability differently and so a single method provides limited explainability capacity. Which is the best method to use? 

The scientific data are of different modalities and much more diverse than the image or text bencmarking datasets used in the Computer Vision ot Natural Language Processing research communities, where the explainability methods originate from. 

* **A single DNN format/framework/architecture is supported**

Lucid supports only [Tensoflow](https://www.tensorflow.org/), Captum is the library for [PyTorch](https://pytorch.org/) users, [iNNvstigate](https://github.com/albermax/innvestigate) has been used by Keras users. Pytorch-grad-cam supports a single method for a single format and Pytorch-cnn-visualizaiton even limits the choice to a single DNN type- CNN. While this is not an issue for the current most popular framework communities, not all are supported by mature libraries and most importantly are not "future-proof", e.g. [Caffe](https://caffe.berkeleyvision.org/) has been the most polular framework by the computer vision comminuty in 2018, while now is the era of Tensorflow and Pytorch.  

* **The choice of supported XAI methods seems random**

ELI5, for example, supports multiple frameworks/formats and XAI methods, but is not clear how the selection of these methods has been made. Futhermore the library is not maintained since 2020 and hence lacks any methods proposed since. Again, which are the best methods to use?

* **Requires (X)AI expertise**
  
For example the DeepVis Toolbox requires DNN knowledge and is used only by AI experts mostly within the computer vision community. 

## Scientific communities and non (X)AI experts needs

All of the above issues have a common reason: **_the libraries have been developed by (X)AI researchers for the (X)AI researchers_**.
 
Even for the purpose of advancing (X)AI research itself, systematic study of the properties of the relevance scores is lacking and currently in the XAI literature often the human interpretation is intertwined with the one of the explainer. Suitable datasets for such a study are lacking: the popular and simplest dataset used in publicaitons,  MNIST, is too complex for such a prurpose with 10 classes and no structural content variation. The XAI community does not publish work on simple scientific datasets, which would make the technology understandable and trustworthy by non-expert scientists on intuitive level. 

Another important show-stopper for *reproducible AI-enabled science* is the plethora of current formats and the speed in which they become obsolete. Some libraries are also becoming obsolete for the lack research funding, being essential for their creation (e.g. iNNvestigate).


## DIANNA for Open Science and non (X)AI experts

DIANNA is designed to address the above challenges: **_the library is developed by the Research Software enigneers with XAI knowledge for scientists: both the X(AI) and especially not X(AI) experts_**. We have strived to create well-documented, well-tested software, conforming with the modern software engineering practices.

 
*	**Systematically chosen generic explainers**

DIANNA includes few XAI approaches which have been evaluated using [systematically defined criteria](https://arxiv.org/ftp/arxiv/papers/1912/1912.05100.pdf). Using a relevant subset of these criteria, we have included complementary state-of-the-art XAI methods: [LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf), [RISE](http://bmvc2018.org/contents/papers/1064.pdf) and [KernelSHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). These methods are model architecture agnostic.

* **Simple unified interface**
  
DIANNA provided a simple unified interface to the chosen XAI methods (explainers). The library can be used both from the command-line and we are working on an easy to use [dashboard](https://github.com/dianna-ai/dianna/tree/main?tab=readme-ov-file#dashboard), which can already be run with some of our examples for demo purposes.

*	**Standard DNN format**

DIANNA is future-proof, the library is conforming with the [Open Neural Network eXchange (ONNX) standart](https://onnx.ai/). Using ONNX is not only very beneficial for interoperability, hence enabling reproducible-science, but ONNX-compatible runtimes and libraries are designed to maximize performance across hardware. 

* **Multiple data modalities** 

Most libraries support at most 2 modalities: images and text. While DIANNA has also started with these two, now it also supports time series and tabular data with embeddings beging work in progress. 

*	**Simple datasets**

To demonstrate the usefulness and enable studying the properties of the relevances on an intuitive level, DIANNA comes with new and existing [datasets](https://github.com/dianna-ai/dianna/tree/main?tab=readme-ov-file#datasets) and [models](https://github.com/dianna-ai/dianna/tree/main?tab=readme-ov-file#onnx-models) for both classification and regression tasks for image, text, time series and tabular data.

* **Scientific use cases** 

DIANNA explicitely shows how to use XAI in AI-enhanced scientific research. [Tutorials](https://github.com/dianna-ai/dianna/tree/main?tab=readme-ov-file#tutorials) for use-cases from the natural, envoronmental and social sciences are provided. Some of the explanability examples on both simple and scientific use-cases are also presented within the dashboard.

