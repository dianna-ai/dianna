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

        tokenized_sentences = []
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
            tokenized_sentences.append(tokens_numerical)

        # run the model, applying a sigmoid because the model outputs logits
        logits = self.run_model(tokenized_sentences)
        pred = np.apply_along_axis(sigmoid, 1, logits)

        # output pos/neg
        positivity = pred[:, 0]
        negativity = 1 - positivity
        return np.transpose([negativity, positivity])
