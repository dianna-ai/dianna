<!-- this file contains the Statement of Need for JOSS -->

# Why DIANNA?

There are numerous other python OSS [eXplainable AI (XAI) libraries](https://github.com/wangyongjie-ntu/Awesome-explainable-AI#python-librariessort-in-alphabeta-order), the Awsome explainable AI features 57 of them, of which 9 have more than 2000 github stars:
[SHAP](https://github.com/slundberg/shap)![](https://img.shields.io/github/stars/slundberg/shap.svg?style=social),
[LIME](https://github.com/marcotcr/lime) ![](https://img.shields.io/github/stars/marcotcr/lime.svg?style=social),
[pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)![](https://img.shields.io/github/stars/utkuozbulak/pytorch-cnn-visualizations?style=social),
[InterpretML](https://github.com/interpretml/interpret) ![](https://img.shields.io/github/stars/InterpretML/interpret.svg?style=social)
[Lucid](https://github.com/tensorflow/lucid) ![](https://img.shields.io/github/stars/tensorflow/lucid.svg?style=social),
[Pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)![](https://img.shields.io/github/stars/jacobgil/pytorch-grad-cam?style=social),
[Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox) ![](https://img.shields.io/github/stars/yosinski/deep-visualization-toolbox?style=social),
[Captum](https://github.com/pytorch/captum) ![](https://img.shields.io/github/stars/pytorch/captum.svg?style=social), 
[ELI5](https://github.com/TeamHG-Memex/eli5) ![](https://img.shields.io/github/stars/TeamHG-Memex/eli5.svg?style=social), etc.

These libraries currently have serious limitaitons in respect to the usage by the varoius scientific communities:

* **A single XAI method or single data modality is implemented**

While SHAP, LIME, Pytorch-grad-cam, DeepLIFT, etc. enjoy enormous popularity, the specific methods are not always suitable for the research task and/or data modality (e.g. Grad cam is applicable only for images). Most importantly, different methods approach the AI explainability differently and so a single method provides limited explainability capacity. Which is the best method to use?

* **A single DNN format/framework/architecture is supported**

Lucid supports only [Tensoflow](https://www.tensorflow.org/), Captum is the library for [PyTorch](https://pytorch.org/) users, [iNNvstigate](https://github.com/albermax/innvestigate) has been used by Keras users. Pytorch-grad-cam supports a single method for a single format and Pytorch-cnn-visualizaiton even limits the choice to a single DNN type- CNN. While this is not an issue for the current most popular framework communities, not all are supported by mature libraries (e.g. ) and most importantly are not "future-proof", e.g. [Caffe](https://caffe.berkeleyvision.org/) has been the most polular framework by the computer vision comminuty in 2018, while now is the era of Tensorflow, Pytorch and Keras.  

* **The choice of supported XAI methods seems random**

ELI5, for example, supports multiple frameworks/formats and XAI methods, but is not clear how the selection of these methods has been made. Futhermore the library is not maintained since 2020 and hence lacks any methods proposed since. Again, which are the best methods to use?

* **Requires (X)AI expertise**
For example the DeepVis Toolbox requires DNN knowledge and is used only by AI experts mostly within the computer vision community (which is large and can explain the number of stars). 

All of the above issues have a common reason: _the libraries have been developed by (X)AI reseachers for the (X)AI reseachrs_.
 
Even for the purpose of advancing (X)AI research, systematic study of the properties of the relevance "heatmaps" is lacking and currently in the XAI literature often the human interpretation is intertwined with the one of the exlainer. Suitable datasets for such a study are lacking: the popular and simplest dataset used in publicaitons,  MNIST, is too complex for such a prurpose with 10 classes and no structural content variation. The XAI community does not publish work on simple scientific datasets, which would make the technology understandable and trustworthy by non-expert scientists on intuitive level. Another important show-stopper for reproducible AI-enabled science is the plethora of current formats and the speed in which they become obsolete. Some libraries are also becoming obsolete for the lack of support of the research projects which have sponsored their creation (e.g. iNNvestigate).

DIANNA is designed to address the above challenges: _the library is developed by the Research Software enigneers with XAI knowledge for scientists: both the X(AI) and especially not X(AI) experts._ We have strived to create well-documented, well-tested software, conforming with the modern software engineering practices.

*	**Simple datasets**

To demonstrate the usefulness and enable studying the properties of the "heatmaps" on an intuitive level, DIANNA comes with two new datasets: [Simple Geometric](https://doi.org/10.5281/zenodo.5012824)<img width="20" alt="SimpleGeometric Logo" src="https://user-images.githubusercontent.com/3244249/151817429-80f38846-8a4b-4471-a4c9-df7d19b668e5.png">
 and [Simple Scientific (LeafSnap30)](https://doi.org/10.5281/zenodo.5061352)<img width="20" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/151817480-649ad3b7-2b4b-4aa3-a5d6-ebfaa6aa614a.png"> – subset of [LeafSnap dataset](http://leafsnap.com/dataset/). Tree species classification on the LeafSnap data is a good simple scientific example of a problem tackled with both classical Computer Vision in the past and the superior DL method. Enhanced with explaianble tool, the DL approach is a clear winnner. DIANNA also uses 2 well-established bencharks - a (modified to binary) MNIST and the [Stanford Sentimet Treebank](https://nlp.stanford.edu/sentiment/index.html).
 
*	**Systematically chosen generic explainers**

DIANNA includes few XAI approaches which have been evaluated using [systematically defined criteria](https://arxiv.org/ftp/arxiv/papers/1912/1912.05100.pdf). Using a relevant subset of these criteria, we have included complementary state-of-the-art XAI methods: [LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf), [RISE](http://bmvc2018.org/contents/papers/1064.pdf) and [KernelSHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). These methods are model architecture agnostic.

*	**Standart DNN format**

DIANNA is future-proof, the library is conforming with the [Open Neural Network eXchange (ONNX) standart](https://onnx.ai/). Using ONNX is not only very beneficial for interoperability, hence enabling reproducible-science, but ONNX-compatible runtimes and libraries are designed to maximize performance across hardware. 

* **Multiple data modalities** 

Most libraries support at most 2 modalities: images and text. While DIANNA has also started with these two, there are plans to include modalities, many scientists lack support for: time series, tabular data and embeddings. 

