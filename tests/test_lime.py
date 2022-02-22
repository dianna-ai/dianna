from unittest import TestCase
import numpy as np
import dianna
import dianna.visualization
from dianna.methods.lime import LIME
from tests.test_onnx_runner import generate_data
from tests.utils import ModelRunner
from tests.utils import run_model


class LimeOnImages(TestCase):
    """Suite of Lime tests for the image case."""
    def test_lime_function(self):
        """Test if lime runs and outputs are correct given some data and a model function."""
        np.random.seed(42)
        input_data = np.random.random((224, 224, 3))
        labels = ('y', 'x', 'channels')
        heatmap_expected = np.load('tests/test_data/heatmap_lime_function.npy')

        explainer = LIME(random_state=42, axis_labels=labels)
        heatmap = explainer.explain_image(run_model, input_data, num_samples=100)

        assert heatmap[0].shape == input_data.shape[:2]
        assert np.allclose(heatmap, heatmap_expected, atol=.01)

    def test_lime_filename(self):
        """Test if lime runs and outputs are correct given some data and a model file."""
        np.random.seed(42)
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        labels = ('channels', 'y', 'x')

        heatmap = dianna.explain_image(model_filename, input_data, method="LIME", random_state=42,
                                       axis_labels=labels)

        heatmap_expected = np.load('tests/test_data/heatmap_lime_filename.npy')
        assert heatmap[0].shape == input_data[0].shape
        assert np.allclose(heatmap, heatmap_expected, atol=.01)


def test_lime_text():
    """Tests exact expected output given a text and model for Lime."""
    model_path = 'tests/test_data/movie_review_model.onnx'
    word_vector_file = 'tests/test_data/word_vectors.txt'
    runner = ModelRunner(model_path, word_vector_file, max_filter_size=5)
    review = 'such a bad movie'
    expected_words = ['bad', 'such', 'movie', 'a']
    expected_word_indices = [7, 0, 11, 5]
    expected_scores = [.492, -.046, .036, -.008]

    explanation = dianna.explain_text(runner, review, labels=[0], method='LIME', random_state=42)[0]
    words = [element[0] for element in explanation]
    word_indices = [element[1] for element in explanation]
    scores = [element[2] for element in explanation]

    assert words == expected_words
    assert word_indices == expected_word_indices
    assert np.allclose(scores, expected_scores, atol=.01)
