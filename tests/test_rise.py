from unittest import TestCase
import numpy as np
import dianna
import dianna.visualization
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
        model_filename = 'tests/test_data/mnist_model.onnx'

        # explainer = rise.RISE()
        data = get_mnist_1_data()
        heatmaps = dianna.explain_image(model_filename, data, method="RISE", n_masks=200)
        # p_keep = explainer._determine_p_keep_for_images(data, 50, get_function(model_filename))

        # print(p_keep)
        # assert False

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
