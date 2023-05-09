from unittest import TestCase
import numpy as np
import pytest
import dianna
import dianna.visualization
from dianna.methods.lime import LIMEImage
from tests.test_onnx_runner import generate_data
from tests.utils import assert_explanation_satisfies_expectations
from tests.utils import load_movie_review_model
from tests.utils import run_model


class LimeOnImages(TestCase):
    """Suite of Lime tests for the image case."""

    @staticmethod
    def test_lime_function():
        """Test if lime runs and outputs are correct given some data and a model function."""
        input_data = np.random.random((224, 224, 3))
        heatmap_expected = np.load('tests/test_data/heatmap_lime_function.npy')
        labels = [1]

        explainer = LIMEImage(random_state=42)
        heatmap = explainer.explain(run_model,
                                    input_data,
                                    labels,
                                    num_samples=100)

        assert heatmap[0].shape == input_data.shape[:2]
        assert np.allclose(heatmap, heatmap_expected, atol=1e-5)

    @staticmethod
    def test_lime_filename():
        """Test if lime runs and outputs are correct given some data and a model file."""
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        axis_labels = ('channels', 'y', 'x')
        labels = [1]

        heatmap = dianna.explain_image(model_filename,
                                       input_data,
                                       method='LIME',
                                       labels=labels,
                                       random_state=42,
                                       axis_labels=axis_labels)

        heatmap_expected = np.load('tests/test_data/heatmap_lime_filename.npy')
        assert heatmap[0].shape == input_data[0].shape
        assert np.allclose(heatmap, heatmap_expected, atol=1e-5)

    def setUp(self) -> None:
        """Set seed."""
        np.random.seed(42)


class LimeOnText(TestCase):
    """Suite of Lime tests for the text case."""

    def test_lime_text(self):
        """Tests exact expected output given a text and model for Lime."""
        review = 'such a bad movie'
        expected_words = ['bad', 'such', 'movie', 'a']
        expected_word_indices = [2, 0, 3, 1]
        expected_scores = [0.49226245, -0.04637814, 0.03648112, -0.00837716]

        explanation = dianna.explain_text(self.runner,
                                          review,
                                          tokenizer=self.runner.tokenizer,
                                          labels=[0],
                                          method='LIME',
                                          random_state=42)[0]

        assert_explanation_satisfies_expectations(explanation, expected_scores,
                                                  expected_word_indices,
                                                  expected_words)

    def test_lime_text_special_chars(self):
        """Tests exact expected output given a text with special characters and model for Lime."""
        review = 'such a bad movie "!?\'"'  # fails, assertion error in __call__
        expected_words = ['bad', 'such', 'movie', 'a', '"!?\'"']
        expected_word_indices = [2, 0, 3, 1, 4]
        expected_scores = [
            0.49421639, -0.04616689, 0.04045723, -0.00912872, -0.00148593
        ]

        explanation = dianna.explain_text(self.runner,
                                          review,
                                          tokenizer=self.runner.tokenizer,
                                          labels=[0],
                                          method='LIME',
                                          random_state=42)[0]

        assert_explanation_satisfies_expectations(explanation, expected_scores,
                                                  expected_word_indices,
                                                  expected_words)

    def setUp(self) -> None:
        """Load the movie review model."""
        self.runner = load_movie_review_model()


@pytest.mark.parametrize(
    'text',
    [
        'review with !!!?',  # fails, assertion error in __call__
        'review with! ?',
        'review with???!',  # fails, assertion error in __call__
        'Review with Capital',
        'Hello, to the world!',
    ])
class TestLimeOnTextSpecialCharacters:
    """Regression tests for inputs with symbols for LIME (https://github.com/dianna-ai/dianna/issues/437)."""
    runner = load_movie_review_model(
    )  # load model once for all tests in this class

    def test_lime_text_special_chars_regression_test(self, text):
        """Just don't raise an error on this input with special characters."""
        _ = dianna.explain_text(self.runner,
                                text,
                                tokenizer=self.runner.tokenizer,
                                labels=[0],
                                method='LIME',
                                random_state=0)


@pytest.fixture
def tokenizer():
    from dianna.utils.tokenizers import SpacyTokenizer
    return SpacyTokenizer()


@pytest.mark.parametrize(
    'text',
    [
        'Hello, to the world!',  # [0, 3, 8, 2, 148, 571], length=6, passes
        'HelloUNKWORDZ UNKWORDZ UNKWORDZ UNKWORDZUNKWORDZ',  # [0, 0, 0, 0, 1], length=4
        'UNKWORDZUNKWORDZ UNKWORDZ UNKWORDZ UNKWORDZUNKWORDZ',  # [0, 0, 0, 0, 1], length=4
        'Hello, to the UNKWORDZ!',  # [0, 3, 8, 2, 0, 571], length=6, passes
        'UNKWORDZ, UNKWORDZ the UNKWORDZUNKWORDZ',  # [0, 3, 0, 2, 0], length=5
        'UNKWORDZUNKWORDZ to the UNKWORDZUNKWORDZ',  # [0, 8, 2, 0, 1], length=4
        'UNKWORDZUNKWORDZ to UNKWORDZ UNKWORDZ!',  # [0, 8, 0, 0, 571], length=5
        'Hello, UNKWORDZ the worldUNKWORDZ',  # [0, 3, 0, 2, 0], length=5
        'UNKWORDZUNKWORDZ to UNKWORDZ worldUNKWORDZ',  # [0, 8, 0, 0, 1], length=4
        'UNKWORDZUNKWORDZ UNKWORDZ UNKWORDZ UNKWORDZUNKWORDZ',  # [0, 0, 0, 0, 1], length=4
    ])
def test_spacytokenizer(text, tokenizer):
    tokens = tokenizer.tokenize(text)

    print()
    print(len(tokens), tokens)

    assert len(tokens) == 6
