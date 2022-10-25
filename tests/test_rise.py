from unittest import TestCase

import dianna
import dianna.visualization
import numpy as np
from dianna.methods.rise import RISEImage, RISEText
from dianna.utils import get_function
from tests.utils import ModelRunner, run_model, get_mnist_1_data

from .test_onnx_runner import generate_data


class RiseOnImages(TestCase):
    """Suite of RISE tests for the image case."""
    def test_rise_function(self):
        """Test if rise runs and outputs the correct shape given some data and a model function."""
        input_data = np.random.random((224, 224, 3))
        # y and x axis labels are not actually mandatory for this test
        axis_labels = ['y', 'x', 'channels']

        heatmaps = dianna.explain_image(run_model, input_data, method="RISE", axis_labels=axis_labels, n_masks=200, p_keep=.5)
        heatmaps_expected = np.load('tests/test_data/heatmap_rise_function.npy')

        assert heatmaps[0].shape == input_data.shape[:2]
        assert np.allclose(heatmaps, heatmaps_expected, atol=1e-5)

    def test_rise_filename(self):
        """Test if rise runs and outputs the correct shape given some data and a model file."""
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1).astype(np.float32)[0]
        # y and x axis labels are not actually mandatory for this test
        axis_labels = ['channels', 'y', 'x']

        heatmaps = dianna.explain_image(model_filename, input_data, method="RISE", axis_labels=axis_labels, n_masks=200, p_keep=.5)
        heatmaps_expected = np.load('tests/test_data/heatmap_rise_filename.npy')

        assert heatmaps[0].shape == input_data.shape[1:]
        assert np.allclose(heatmaps, heatmaps_expected, atol=1e-5)

    def test_rise_determine_p_keep_for_images(self):
        """Tests exact expected p_keep given an image and model."""
        np.random.seed(0)
        expected_p_exact_keep = .1
        model_filename = 'tests/test_data/mnist_model.onnx'
        data = get_mnist_1_data().astype(np.float32)

        p_keep = RISEImage()._determine_p_keep(  # pylint: disable=protected-access
            data, get_function(model_filename))

        assert np.isclose(p_keep, expected_p_exact_keep)


class RiseOnText(TestCase):
    """Suite of RISE tests for the text case."""
    def test_rise_text(self):
        """Tests exact expected output given a text and model."""
        np.random.seed(42)
        model_path = 'tests/test_data/movie_review_model.onnx'
        word_vector_file = 'tests/test_data/word_vectors.txt'
        runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)
        review = 'such a bad movie'
        expected_words = ['such', 'a', 'bad', 'movie']
        expected_word_indices = [0, 1, 2, 3]
        expected_positive_scores = [0.3044678, 0.28736606, 0.03623142, 0.23650846]

        positive_explanation = dianna.explain_text(runner, review, tokenizer=runner.tokenizer,
                                                   labels=(1, 0), method='RISE', p_keep=.5)[0]

        words = [element[0] for element in positive_explanation]
        word_indices = [element[1] for element in positive_explanation]
        positive_scores = [element[2] for element in positive_explanation]
        assert words == expected_words
        assert word_indices == expected_word_indices
        assert np.allclose(positive_scores, expected_positive_scores, atol=1e-5)

    def test_rise_determine_p_keep_for_text(self):
        """Tests exact expected p_keep given a text and model."""
        np.random.seed(0)
        expected_p_exact_keep = .5
        model_path = 'tests/test_data/movie_review_model.onnx'
        word_vector_file = 'tests/test_data/word_vectors.txt'
        runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)
        input_text = 'such a bad movie'
        runner = get_function(runner)
        input_tokens = np.asarray(runner.tokenizer.tokenize(input_text))

        p_keep = RISEText()._determine_p_keep(input_tokens, runner, runner.tokenizer)  # pylint: disable=protected-access
        assert np.isclose(p_keep, expected_p_exact_keep)
