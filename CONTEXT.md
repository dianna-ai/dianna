<!-- this file contains the Statement of Need for JOSS -->

# Why DIANNA?

There are numerous other python OSS [XAI libraries](https://github.com/wangyongjie-ntu/Awesome-explainable-AI#python-librariessort-in-alphabeta-order), the Awsome explainable AI features 57 of them, of which 9 have more than 2000 github stars:
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

