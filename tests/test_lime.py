from unittest import TestCase

import numpy as np
import dianna
import dianna.visualization
from tests.test_onnx_runner import generate_data
from tests.utils import ModelRunner


def run_model(input_data):
    n_class = 2
    batch_size = input_data.shape[0]

    return np.random.random((batch_size, n_class))


class LimeOnImages(TestCase):

    def test_lime_function(self):
        # shape is batch, y, x, channel
        input_data = np.random.random((1, 224, 224, 3))

        heatmap = dianna.explain_image(run_model, input_data, method="LIME")

        assert heatmap.shape == input_data[0].shape[:2]

    def test_lime_filename(self):
        model_filename = 'tests/test_data/mnist_model.onnx'
        black_and_white = generate_data(batch_size=1).transpose((0, 3, 2, 1))
        input_data = np.zeros(list(black_and_white.shape[:-1]) + [3]) + black_and_white
        print('shape', input_data.shape)

        heatmaps = dianna.explain_image(model_filename, input_data, method="LIME")

        assert heatmaps[0].shape == input_data[0].shape[:2]


def test_lime_text():
    model_path = 'tests/test_data/movie_review_model.onnx'
    word_vector_file = 'tests/test_data/word_vectors.txt'
    runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)

    review = 'such a bad movie'

    explanation = dianna.explain_text(runner, review, method='LIME', random_state=42)
    words = [element[0] for element in explanation]
    word_indices = [element[1] for element in explanation]
    scores = [element[2] for element in explanation]

    expected_words = ['bad', 'such', 'movie', 'a']
    expected_word_indices = [7, 0, 11, 5]
    expected_scores = [-.492, .046, -.036, .008]
    assert words == expected_words
    assert word_indices == expected_word_indices
    assert np.allclose(scores, expected_scores, atol=.01)
