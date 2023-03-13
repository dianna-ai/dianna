from unittest import TestCase
import numpy as np
import dianna
import dianna.visualization
from dianna.methods.rise import RISEImage
from dianna.methods.rise import RISEText
from dianna.utils import get_function
from tests.utils import assert_explanation_satisfies_expectations
from tests.utils import get_mnist_1_data
from tests.utils import load_movie_review_model
from tests.utils import run_model
from .test_onnx_runner import generate_data


class RiseOnImages(TestCase):
    """Suite of RISE tests for the image case."""

    @staticmethod
    def test_rise_function():
        """Test if rise runs and outputs the correct shape given some data and a model function."""
        input_data = np.random.random((224, 224, 3))
        axis_labels = ['y', 'x', 'channels']
        labels = [1]
        heatmaps_expected = np.load('tests/test_data/heatmap_rise_function.npy')

        heatmaps = dianna.explain_image(run_model, input_data, "RISE", labels, axis_labels=axis_labels, n_masks=200,
                                        p_keep=.5)

        assert heatmaps[0].shape == input_data.shape[:2]
        assert np.allclose(heatmaps, heatmaps_expected, atol=1e-5)

    @staticmethod
    def test_rise_filename():
        """Test if rise runs and outputs the correct shape given some data and a model file."""
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1).astype(np.float32)[0]
        heatmaps_expected = np.load('tests/test_data/heatmap_rise_filename.npy')
        labels = [1]

        heatmaps = dianna.explain_image(model_filename, input_data, "RISE", labels, n_masks=200, p_keep=.5)

        assert heatmaps[0].shape == input_data.shape[1:]
        print(heatmaps_expected.shape)
        assert np.allclose(heatmaps, heatmaps_expected, atol=1e-5)

    @staticmethod
    def test_rise_determine_p_keep_for_images():
        """Tests exact expected p_keep given an image and model."""
        np.random.seed(0)
        expected_p_exact_keep = .4
        model_filename = 'tests/test_data/mnist_model.onnx'
        data = get_mnist_1_data().astype(np.float32)

        p_keep = RISEImage()._determine_p_keep(
            data, get_function(model_filename))

        assert np.isclose(p_keep, expected_p_exact_keep)


class RiseOnText(TestCase):
    """Suite of RISE tests for the text case."""

    def test_rise_text(self):
        """Tests exact expected output given a text and model."""
        review = 'such a bad movie'
        expected_words = ['such', 'a', 'bad', 'movie']
        expected_word_indices = [0, 1, 2, 3]
        expected_positive_scores = [0.30, 0.29, 0.04, 0.25]

        positive_explanation = dianna.explain_text(self.runner, review, tokenizer=self.runner.tokenizer,
                                                   labels=(1, 0), method='RISE', p_keep=.5)[0]

        assert_explanation_satisfies_expectations(positive_explanation, expected_positive_scores, expected_word_indices,
                                                  expected_words)

    def test_rise_determine_p_keep_for_text(self):
        """Tests exact expected p_keep given a text and model."""
        expected_p_exact_keep = .7
        input_text = 'such a bad movie'
        runner = get_function(self.runner)
        input_tokens = np.asarray(runner.tokenizer.tokenize(input_text))

        p_keep = RISEText()._determine_p_keep(input_tokens, runner,  # pylint: disable=protected-access
                                              runner.tokenizer)
        assert np.isclose(p_keep, expected_p_exact_keep)

    def setUp(self) -> None:
        """Set seed and load runner."""
        np.random.seed(0)
        self.runner = load_movie_review_model()
