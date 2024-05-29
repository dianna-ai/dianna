import os
import numpy as np
from _shared import data_directory
from scipy.special import expit as sigmoid
from torchtext.vocab import Vectors
from dianna import utils
from dianna.utils.tokenizers import SpacyTokenizer


class MovieReviewsModelRunner:
    """Creates runner for movie review model."""

    def __init__(self, model, word_vectors=None, max_filter_size=5):
        """Initializes the class."""
        if word_vectors is None:
            word_vectors = data_directory / 'movie_reviews_word_vectors.txt'

        self.run_model = utils.get_function(model)
        self.vocab = Vectors(word_vectors, cache=os.path.dirname(word_vectors))
        self.max_filter_size = max_filter_size
        self.tokenizer = SpacyTokenizer()

    def __call__(self, sentences):
        """Call Runner."""
        # ensure the input has a batch axis
        if isinstance(sentences, str):
            sentences = [sentences]

        output = []
        for sentence in sentences:
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

            # run the model, applying a sigmoid because the model outputs logits, remove any remaining batch axis
            pred = float(sigmoid(self.run_model([tokens_numerical])))
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
