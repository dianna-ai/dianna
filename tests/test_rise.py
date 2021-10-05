from unittest import TestCase

import numpy as np
import dianna
import dianna.visualization
from tests.utils import ModelRunner
from .test_onnx_runner import generate_data


def run_model(input_data):
    n_class = 2
    batch_size = input_data.shape[0]

    return np.random.random((batch_size, n_class))


class rise_on_images(TestCase):

    def test_rise_function(self):
        # shape is batch, y, x, channel
        input_data = np.random.random((1, 224, 224, 3))

        heatmaps = dianna.explain_image(run_model, input_data, method="RISE", n_masks=200)

        assert heatmaps[0].shape == input_data[0].shape[:2]

    def test_rise_filename(self):
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)

        heatmaps = dianna.explain_image(model_filename, input_data, method="RISE", n_masks=200)

        assert heatmaps[0].shape == input_data[0].shape[:2]


class rise_on_text(TestCase):
    def test_rise_text(self):
        # fix the seed for testing
        np.random.seed(42)

        model_path = 'tests/test_data/movie_review_model.onnx'
        word_vector_file = 'tests/test_data/word_vectors.txt'
        runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)

        review = 'such a bad movie'

        explanation = dianna.explain_text(runner, review, method='RISE')
        words = [element[0] for element in explanation]
        word_indices = [element[1] for element in explanation]
        negative_scores = [element[2] for element in explanation]
        positive_scores = [element[3] for element in explanation]

        expected_words = ['such', 'a', 'bad', 'movie']
        expected_word_indices = [0, 5, 7, 11]
        expected_negative_scores = [0.6504735, 0.6498708, 1.0003519, 0.65321875]
        expected_positive_scores = [0.3295266, 0.3521292, 0.023648001, 0.3347813]

        assert words == expected_words
        assert word_indices == expected_word_indices
        assert np.allclose(positive_scores, expected_positive_scores)
        assert np.allclose(negative_scores, expected_negative_scores)
