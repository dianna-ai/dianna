from unittest import TestCase

import dianna
import dianna.visualization
import numpy as np
from dianna.methods import rise
from dianna.utils import get_function
from tests.utils import ModelRunner, run_model, get_mnist_1_data

from .test_onnx_runner import generate_data


def make_channels_first(data):
    return data.transpose((0, 3, 1, 2))


class RiseOnImages(TestCase):

    def test_rise_function(self):
        input_data = np.random.random((1, 224, 224, 3))
        # y and x axis labels are not actually mandatory for this test
        axes_labels = ['batch', 'y', 'x', 'channels']

        heatmaps = dianna.explain_image(run_model, input_data, method="RISE", axes_labels=axes_labels, n_masks=200)

        assert heatmaps[0].shape == input_data[0].shape[:2]

    def test_rise_filename(self):
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1).astype(np.float32)
        # y and x axis labels are not actually mandatory for this test
        axes_labels = ['batch', 'channels', 'y', 'x']

        heatmaps = dianna.explain_image(model_filename, input_data, method="RISE", axes_labels=axes_labels, n_masks=200)

        assert heatmaps[0].shape == input_data[0].shape[1:]

    def test_rise_determine_p_keep_for_images(self):
        """
        When using the large sample size of 10000, the mean STD for each class for the following p_keeps
        [     0.1,       0.2,      0.3,       0.4,       0.5,       0.6,       0.7,      0.8,      0.9]
        is as follows:
        [2.069784, 2.600222, 2.8940516, 2.9950087, 2.9579144, 2.8919978, 2.6288269, 2.319147, 1.763127]
        So best p_keep should be .4 or .5 ( or at least between .3 and .6).
        """
        np.random.seed(0)
        expected_p_keeps = [.3, .4, .5, .6]
        expected_p_exact_keep = .4
        model_filename = 'tests/test_data/mnist_model.onnx'
        data = get_mnist_1_data().astype(np.float32)

        p_keep = rise.RISE()._determine_p_keep_for_images(  # pylint: disable=protected-access
            data, get_function(model_filename))

        assert p_keep in expected_p_keeps  # Sanity check: is the outcome in the acceptable range?
        assert p_keep == expected_p_exact_keep  # Exact test: is the outcome the same as before?


class RiseOnText(TestCase):
    def test_rise_text(self):
        np.random.seed(42)
        model_path = 'tests/test_data/movie_review_model.onnx'
        word_vector_file = 'tests/test_data/word_vectors.txt'
        runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)
        review = 'such a bad movie'

        positive_explanation = dianna.explain_text(runner, review, labels=(1, 0), method='RISE')[0]
        words = [element[0] for element in positive_explanation]
        word_indices = [element[1] for element in positive_explanation]
        positive_scores = [element[2] for element in positive_explanation]

        expected_words = ['such', 'a', 'bad', 'movie']
        expected_word_indices = [0, 5, 7, 11]
        expected_positive_scores = [0.3295266, 0.3521292, 0.023648001, 0.3347813]

        assert words == expected_words
        assert word_indices == expected_word_indices
        assert np.allclose(positive_scores, expected_positive_scores)

    def test_rise_determine_p_keep_for_text(self):
        '''
        When using the large sample size of 10000, the mean STD for each class for the following p_keeps
        [       0.1,      0.2,        0.3,        0.4,        0.5,        0.6,       0.7,       0.8,       0.9]
        is as follows:
        [0.18085817, 0.239386, 0.27801532, 0.30555934, 0.31592548, 0.31345606, 0.2901688, 0.2539522, 0.19383237]
        So best p_keep should be .4 or .5 ( or at least between .4 and .7).
        '''
        np.random.seed(0)
        expected_p_keeps = [.3, .4, .5, .6]
        expected_p_exact_keep = .5
        model_path = 'tests/test_data/movie_review_model.onnx'
        word_vector_file = 'tests/test_data/word_vectors.txt'
        runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)
        input_text = 'such a bad movie'
        runner = get_function(runner)
        input_tokens = np.asarray(runner.tokenizer(input_text))

        p_keep = rise.RISE()._determine_p_keep_for_text(input_tokens, runner)  # pylint: disable=protected-access

        assert p_keep in expected_p_keeps  # Sanity check: is the outcome in the acceptable range?
        assert p_keep == expected_p_exact_keep  # Exact test: is the outcome the same as before?
