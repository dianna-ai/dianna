import numpy as np
from keras import utils as keras_utils
from PIL import Image
from PIL import ImageStat
from dianna.utils import move_axis
from dianna.utils import to_xarray


def preprocess_img_resnet(path):
    """Resnet specific function for preprocessing.

    Reshape figure to 224,224 and get colour channel at position 0.
    Also: for resnet preprocessing: normalize the data. This works specifically for ImageNet.
    See: https://github.com/onnx/models/tree/main/vision/classification/resnet
    """
    img = keras_utils.load_img(path, target_size=(224, 224))
    img_data = keras_utils.img_to_array(img)
    if img_data.shape[0] != 3:
        # Colour channel is not in position 0; reshape the data
        xarray = to_xarray(img_data, {0: 'height', 1: 'width', 2: 'channels'})
        reshaped_data = move_axis(xarray, 'channels', 0)
        img_data = np.array(reshaped_data)

    # definitions for normalisation (for ImageNet)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])

    norm_img_data = np.zeros(img_data.shape).astype('float32')

    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the values by 255 ([0,1]), and normalize
        # using mean and standard deviation from values above
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 -
                                  mean_vec[i]) / stddev_vec[i]

    return norm_img_data, img


def open_image(file):
    """Open an image from a file and returns it as a numpy array."""
    im = Image.open(file).convert('RGB')
    stat = ImageStat.Stat(im)
    im = np.asarray(im).astype(np.float32)

    if sum(stat.sum
           ) / 3 == stat.sum[0]:  # check the avg with any element value
        return np.expand_dims(im[:, :, 0], axis=2) / 255, im  # if grayscale
    else:
        # else it's colour, reshape to 224x224x3 for resnet
        return preprocess_img_resnet(file)
