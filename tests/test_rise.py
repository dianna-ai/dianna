from unittest import TestCase
import numpy as np
import dianna
import dianna.visualization
from tests.utils import ModelRunner, run_model
from .test_onnx_runner import generate_data


def make_channels_first(data):
    return data.transpose((0, 3, 1, 2))


class RiseOnImages(TestCase):

    def test_rise_function(self):
        # shape is batch, y, x, channel
        input_data = np.random.random((1, 224, 224, 3))
        axes_labels = {0: 'batch', -1: 'channels'}

        heatmaps = dianna.explain_image(run_model, input_data, method="RISE", axes_labels=axes_labels, n_masks=200)

        assert heatmaps[0].shape == input_data[0].shape[:2]

    def test_rise_filename(self):
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1).astype(np.float32)
        axes_labels = {0: 'batch', 1: 'channels'}

        heatmaps = dianna.explain_image(model_filename, input_data, method="RISE", axes_labels=axes_labels, n_masks=200)

        assert heatmaps[0].shape == input_data[0].shape[1:]


class RiseOnText(TestCase):
    def test_rise_text(self):
        np.random.seed(42)

        model_path = 'tests/test_data/movie_review_model.onnx'
        word_vector_file = 'tests/test_data/word_vectors.txt'
        runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)

        review = 'such a bad movie'

        positive_explanation = dianna.explain_text(runner, review, labels=(1, 0), method='RISE')[0]
        print(positive_explanation)
        words = [element[0] for element in positive_explanation]
        word_indices = [element[1] for element in positive_explanation]
        positive_scores = [element[2] for element in positive_explanation]

        expected_words = ['such', 'a', 'bad', 'movie']
        expected_word_indices = [0, 5, 7, 11]
        expected_positive_scores = [0.3295266, 0.3521292, 0.023648001, 0.3347813]

        assert words == expected_words
        assert word_indices == expected_word_indices
        assert np.allclose(positive_scores, expected_positive_scores)
