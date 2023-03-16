"""write all class labels for the ResNet50 model to a txt file"""
import numpy as np
from tensorflow.keras.applications.resnet50 import decode_predictions

class_names = [decode_predictions(np.eye(1,1000, i))[0][0][1] for i in range(0, 1000)]

with open('labels_resnet50.txt', 'w', encoding="utf-8") as file:
    file.write('\n'.join(class_names))
