from unittest import TestCase
import numpy as np
import dianna
import dianna.visualization
from dianna.methods.lime import LIMEImage
from tests.test_onnx_runner import generate_data
from tests.utils import ModelRunner
from tests.utils import run_model


class LimeOnImages(TestCase):
    """Suite of Lime tests for the image case."""
    def test_lime_function(self):
        """Test if lime runs and outputs are correct given some data and a model function."""
        np.random.seed(42)
        input_data = np.random.random((224, 224, 3))
        heatmap_expected = np.load('tests/test_data/heatmap_lime_function.npy')
        labels = [1]

        explainer = LIMEImage(random_state=42)
        heatmap = explainer.explain(run_model, input_data, labels, num_samples=100)

        assert heatmap[0].shape == input_data.shape[:2]
        assert np.allclose(heatmap, heatmap_expected, atol=1e-5)

    def test_lime_filename(self):
        """Test if lime runs and outputs are correct given some data and a model file."""
        np.random.seed(42)
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        axis_labels = ('channels', 'y', 'x')
        labels = [1]

        heatmap = dianna.explain_image(model_filename, input_data, method="LIME", labels=labels, random_state=42,
                                       axis_labels=axis_labels)

        heatmap_expected = np.load('tests/test_data/heatmap_lime_filename.npy')
        assert heatmap[0].shape == input_data[0].shape
        assert np.allclose(heatmap, heatmap_expected, atol=1e-5)


def test_lime_text():
    """Tests exact expected output given a text and model for Lime."""
    model_path = 'tests/test_data/movie_review_model.onnx'
    word_vector_file = 'tests/test_data/word_vectors.txt'
    runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)
    review = 'such a bad movie'
    expected_words = ['bad', 'such', 'movie', 'a']
    expected_word_indices = [2, 0, 3, 1]
    expected_scores = [0.49226245, -0.04637814, 0.03648112, -0.00837716]

    explanation = dianna.explain_text(runner, review, tokenizer=runner.tokenizer,
                                      labels=[0], method='LIME', random_state=42)[0]
    words = [element[0] for element in explanation]
    word_indices = [element[1] for element in explanation]
    scores = [element[2] for element in explanation]

    assert words == expected_words
    assert word_indices == expected_word_indices
    assert np.allclose(scores, expected_scores, atol=1e-5)


def test_lime_text_special_chars():
    """Tests exact expected output given a text with special characters and model for Lime."""
    model_path = 'tests/test_data/movie_review_model.onnx'
    word_vector_file = 'tests/test_data/word_vectors.txt'
    runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)
    review = 'such a bad movie "!?\'"'
    expected_words = ['bad', 'such', 'movie', 'a', '"!?\'"']
    expected_word_indices = [2, 0, 3, 1, 4]
    expected_scores = [0.49421639, -0.04616689, 0.04045723, -0.00912872, -0.00148593]

    explanation = dianna.explain_text(runner, review, tokenizer=runner.tokenizer,
                                      labels=[0], method='LIME', random_state=42)[0]
    print(f'{len(explanation)=}')
    words = [element[0] for element in explanation]
    word_indices = [element[1] for element in explanation]
    scores = [element[2] for element in explanation]

    assert words == expected_words
    assert word_indices == expected_word_indices
    assert np.allclose(scores, expected_scores, atol=1e-5)
