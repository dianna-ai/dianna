import numpy as np
import onnxruntime as ort
import spacy
import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import expit
from torchtext.data import get_tokenizer
from torchtext.vocab import Vectors


def run_model(input_data):
    """
    Simulate a model that outputs 2-classes.
    Args:
        input_data:

    Returns:

    """
    n_class = 2
    batch_size = input_data.shape[0]

    np.random.seed(42)
    return np.random.random((batch_size, n_class))


class ModelRunner():
    def __init__(self, model_path, word_vector_file, max_filter_size):
        self.filename = model_path
        # ensure the spacy english is downloaded
        spacy.cli.download('en_core_web_sm')
        self.tokenizer = get_tokenizer('spacy', 'en_core_web_sm')
        self.vocab = Vectors(word_vector_file, cache='.')

        self.max_filter_size = max_filter_size

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        output = []

        sess = ort.InferenceSession(self.filename)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        for sentence in sentences:
            # get tokens
            tokens = self.tokenizer(sentence)
            if len(tokens) < self.max_filter_size:
                tokens += ['<pad>'] * (self.max_filter_size - len(tokens))

            # numericalize
            tokens = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>'] for token in
                      tokens]
            # feed to model
            onnx_input = {input_name: [tokens]}
            pred = expit(sess.run([output_name], onnx_input)[0])

            # output 2 classes
            positivity = pred[:, 0]
            negativity = 1 - positivity

            output.append(np.transpose([negativity, positivity])[0])

        return np.array(output)


class MnistNet(nn.Module):
    """
    Model designed for mnist. This class works with the load_torch_model
    function for testing deeplift in dianna.
    """

    def __init__(self, kernels=[16, 32], dropout=0.1, classes=2):
        '''
        Two layer CNN model with max pooling.
        '''
        super(MnistNet, self).__init__()
        self.kernels = kernels
        # 1st layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout()
        )
        # 2nd layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1],
                      kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout()
        )
        # pixel 28 / maxpooling 2 * 2 = 7
        self.fc1 = nn.Linear(7 * 7 * kernels[-1], kernels[-1])
        self.fc2 = nn.Linear(kernels[-1], classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def load_torch_model(path_to_model):
    """
    Load pytorch model
    Args:
        path_to_model (str):
    Returns:
        pytorch model
    """
    # create the structure of the model
    # hyper-parameters
    kernels = [16, 32]
    dropout = 0.5
    classes = 2
    # create model
    model = MnistNet(kernels, dropout, classes)
    # load whole model state
    checkpoint = torch.load(path_to_model)
    # load model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model
