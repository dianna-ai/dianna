import numpy as np
import onnxruntime as ort
import spacy
from scipy.special import expit as sigmoid
from torchtext.vocab import Vectors
from dianna.utils.tokenizers import SpacyTokenizer

_mnist_1_data = """
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   38  254 109 0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   87  252 82  0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   135 241 0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   45  244 150 0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   84  254 63  0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   202 223 11  0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   32  254 216 0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   95  254 195 0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   140 254 77  0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   57  237 205 8   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   124 255 165 0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   171 254 81  0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   24  232 215 0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   120 254 159 0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   151 254 142 0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   228 254 66  0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   61  251 254 66  0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   141 254 205 3   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   10  215 254 121 0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   5   198 176 10  0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
"""


def get_mnist_1_data():
    """Gets a single instance (label=1) from the mnist dataset."""
    return np.loadtxt(_mnist_1_data.split()).reshape((1, 1, 28, 28))


def get_dummy_model_function(n_outputs):
    """Create a function that simulates a model that outputs n_outputs.

    Args:
        n_outputs: number of outputs

    Returns:
        dummy model as a function
    """

    def run_model(input_data):
        """Simulate a model that outputs n_outputs.

        Args:
            input_data: input data for the dummy model

        Returns:
            semi random output
        """
        batch_size = input_data.shape[0]

        np.random.seed(42)
        return np.random.random((batch_size, n_outputs))

    return run_model


class ModelRunner:
    """Example model runner for text models used for automated testing."""

    def __init__(self, model_path, word_vector_file, max_filter_size):
        """Initializes the model runner.

        Args:
            model_path: path to the model file
            word_vector_file: path to the vector file
            max_filter_size: maximum filter size of the model
        """
        self.filename = model_path
        # ensure the spacy english is downloaded
        model = 'en_core_web_sm'
        if not spacy.util.is_package(model):
            spacy.cli.download(model)
        self.tokenizer = SpacyTokenizer()
        self.vocab = Vectors(word_vector_file, cache='.')

        self.max_filter_size = max_filter_size

    def __call__(self, sentences):
        """Call Runner."""
        # ensure the input has a batch axis
        if isinstance(sentences, str):
            sentences = [sentences]

        sess = ort.InferenceSession(self.filename)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        output = []
        for sentence in sentences:
            # tokenize and pad to minimum length
            tokens = self.tokenizer.tokenize(sentence.lower())
            if len(tokens) < self.max_filter_size:
                tokens += ['<pad>'] * (self.max_filter_size - len(tokens))

            # numericalize the tokens
            tokens_numerical = [
                self.vocab.stoi[token]
                if token in self.vocab.stoi else self.vocab.stoi['<unk>']
                for token in tokens
            ]

            # run the model, applying a sigmoid because the model outputs logits, remove any remaining batch axis
            onnx_input = {input_name: [tokens_numerical]}
            logits = sess.run([output_name], onnx_input)[0]
            pred = float(sigmoid(logits))
            output.append(pred)

        # output two classes
        positivity = np.array(output)
        negativity = 1 - positivity
        return np.transpose([negativity, positivity])

    def tokenize(self, sentence: str):
        """Tokenize sentence."""
        # tokenize and pad to minimum length
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) < self.max_filter_size:
            tokens += ['<pad>'] * (self.max_filter_size - len(tokens))

        # numericalize the tokens
        tokens_numerical = [
            self.vocab.stoi[token]
            if token in self.vocab.stoi else self.vocab.stoi['<unk>']
            for token in tokens
        ]
        return tokens_numerical


def assert_explanation_satisfies_expectations(explanation, expected_scores,
                                              expected_word_indices,
                                              expected_words):
    """Asserts that the explanation contains the expected values."""
    words = [element[0] for element in explanation]
    word_indices = [element[1] for element in explanation]
    scores = [element[2] for element in explanation]

    assert words == expected_words, f'{words} not equal to expected {expected_words}'
    assert word_indices == expected_word_indices, f'{word_indices} not equal to expected {expected_word_indices}'
    assert np.allclose(scores, expected_scores, atol=1e-2)


def load_movie_review_model():
    """Loads the movie review model."""
    model_path = 'tests/test_data/movie_review_model.onnx'
    word_vector_file = 'tests/test_data/word_vectors.txt'
    return ModelRunner(model_path, word_vector_file, max_filter_size=5)
