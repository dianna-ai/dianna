How to use DIANNA
-----------------

To use DIANNA you need a *trained AI model* (in ONNX format) and a *data
item* (e.g. an image or text, etc.) for which you would like to explain
the output of the model. DIANNA calls an explainable AI method to
produce a “heatmap” of the relevances of each data pont (e.g. pixel,
word) to a given model’s decision overlaid on the data item.

For example usage see the DIANNA `tutorials <././tutorials>`__. For
creating or converting a trained model to ONNX see the **ONNX models**
and for example datasets- the **Datasets** sections below.

.. figure:: https://user-images.githubusercontent.com/3244249/152557189-3ed6fe1a-b461-4cc8-bd2e-e420ee46c784.png
   :alt: Architecture_high_level_resized

   Architecture_high_level_resized
