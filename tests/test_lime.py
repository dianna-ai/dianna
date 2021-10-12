import numpy as np
import dianna
import dianna.visualization
from tests.utils import ModelRunner


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
