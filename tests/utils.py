import numpy as np
import onnxruntime as ort
import spacy
from scipy.special import expit
from torchtext.data import get_tokenizer
from torchtext.vocab import Vectors


def run_model(input_data):
    """
    Simulate a model that outputs 2-classes.

    Args:
        input_data:

    Returns:
        semi random output
    """
    n_class = 2
    batch_size = input_data.shape[0]

    np.random.seed(42)
    return np.random.random((batch_size, n_class))


class ModelRunner:
    """Example model runner for text models used for automated testing."""

    def __init__(self, model_path, word_vector_file, max_filter_size):
        """
        Initializes the model runner.

        Args:
            model_path: path to the model file
            word_vector_file: path to the vector file
            max_filter_size: maximum filter size of the model
        """
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
