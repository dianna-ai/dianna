from unittest import TestCase
import numpy as np
import dianna
import dianna.visualization
import pytest
from dianna.methods import rise
from dianna.utils import get_function
from dianna.utils.onnx_runner import SimpleModelRunner
from tests.utils import ModelRunner, run_model, get_mnist_1_data
from .test_onnx_runner import generate_data


class RiseOnImages(TestCase):

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


    def test_rise_determine_p_keep_for_images(self):
        '''
        When using the large sample size of 10000, the mean STD for each class for the following p_keeps
        [     0.1,       0.2,      0.3,       0.4,       0.5,       0.6,       0.7,       0.8]
        is as follows:
        [2.071084, 2.6516771, 2.896659, 2.9460478, 2.9888847, 2.8803914, 2.6940017, 2.3410206]
        So best p_keep should be .4 or .5 ( or at least between .3 and .6).

        When using 20 n_masks we got this p_keep histogram: [ 1  7 19 24 21 18  8  2]
        When using 30 n_masks we got this p_keep histogram: [ 0  4 11 30 23 23  9]
        When using 50 n_masks we got this p_keep histogram: [ 0  3 16 35 28 14  4]
        When using 100 n_masks we got this p_keep histogram: [ 0  3 14 37 23 21  2]
        When using 200 n_masks we got this p_keep histogram: [ 0  0 16 37 32 15]
        It seems 20 is not enough to have a good chance of getting a good p_keep. For 200 every sample returns a reasonable p_keep but is a bit much to be practicle. I think we should use 100 to be on the save side.

        Returns:

        '''
        np.random.seed(0)
        expected_p_keeps = [.3, .4, .5, .6]
        expected_p_exact_keep = .4
        model_filename = 'tests/test_data/mnist_model.onnx'
        data = get_mnist_1_data()

        explainer = rise.RISE()
        p_keep = explainer._determine_p_keep_for_images(data, get_function(model_filename))

        assert p_keep in expected_p_keeps  # Sanity check: is the outcome in the acceptable range?
        assert p_keep == expected_p_exact_keep  ## Exact test: is the outcome the same as before?
        assert False  # assert some sensible p_keep value (or range)

class RiseOnText(TestCase):
    @pytest.mark.skip
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
