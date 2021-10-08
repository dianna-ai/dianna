from unittest import TestCase

import numpy as np
import dianna
import dianna.visualization
from dianna.methods import LIME
from tests.utils import ModelRunner


def run_model(input_data):
    n_class = 2
    batch_size = input_data.shape[0]

    np.random.seed(42)
    return np.random.random((batch_size, n_class))


class LimeOnImages(TestCase):

    def test_lime_function(self):
        # shape is batch, y, x, channel
        np.random.seed(42)
        input_data = np.random.random((1, 224, 224, 3))

        explainer = LIME(random_state=42)
        heatmap = explainer.explain_image(run_model, input_data, num_samples=100)

        assert heatmap.shape == input_data[0].shape[:2]
        heatmap_expected = np.load('tests/test_data/heatmap_lime_function.npy')
        assert np.allclose(heatmap, heatmap_expected, atol=.01)


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
