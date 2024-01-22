"""Unit tests for LIME text."""
from unittest import TestCase
import pytest
import dianna
import dianna.visualization
from tests.utils import assert_explanation_satisfies_expectations
from tests.utils import load_movie_review_model


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
        review = 'such a bad movie "!?\'"'
        expected_words = ['bad', '?', '!', 'movie', 'such', 'a', "'", '"', '"']
        expected_word_indices = [2, 6, 5, 3, 0, 1, 7, 4, 8]
        expected_scores = [
            0.50032869, 0.06458735, -0.05793979, 0.01413776, -0.01246357,
            -0.00528022, 0.00305347, 0.00185159, -0.00165128
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


@pytest.mark.parametrize('text', [
    'review with !!!?',
    'review with! ?',
    'review with???!',
    'Review with Capital',
    'Hello, to the world!',
    "Let's try something new!",
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
    """Return tokenizer."""
    from dianna.utils.tokenizers import SpacyTokenizer
    return SpacyTokenizer()


@pytest.mark.parametrize(('text', 'length'), [
    ('Hello, to the world!', 6),
    ('HelloUNKWORDZ UNKWORDZ UNKWORDZ UNKWORDZUNKWORDZ', 6),
    ('Hello, to the UNKWORDZ!', 6),
    ('UNKWORDZ, UNKWORDZ the UNKWORDZUNKWORDZ', 6),
    ('Hello, UNKWORDZ the worldUNKWORDZ', 6),
    ('UNKWORDZUNKWORDZ to UNKWORDZ worldUNKWORDZ', 6),
    ('UNKWORDZUNKWORDZ UNKWORDZ UNKWORDZ UNKWORDZUNKWORDZ', 6),
    ('such a bad movie "!?\'"', 9),
    ('such UNKWORDZ UNKWORDZ UNKWORDZ "UNKWORDZUNKWORDZUNKWORDZUNKWORDZ', 9),
    ('UNKWORDZ UNKWORDZ UNKWORDZ movie "!?UNKWORDZ"', 9),
    ('UNKWORDZ UNKWORDZ UNKWORDZ UNKWORDZ UNKWORDZUNKWORDZUNKWORDZ\'UNKWORDZ',
     9),
    ('such UNKWORDZ bad UNKWORDZ UNKWORDZ!UNKWORDZUNKWORDZ"', 9),
    ('UNKWORDZ a UNKWORDZ UNKWORDZ UNKWORDZUNKWORDZ?UNKWORDZUNKWORDZ', 9),
    ('UNKWORDZ a bad UNKWORDZ UNKWORDZ!?\'"', 9),
    ('such UNKWORDZ UNKWORDZ movie "UNKWORDZUNKWORDZ\'UNKWORDZ', 9),
    ('such a bad UNKWORDZ UNKWORDZ!UNKWORDZ\'UNKWORDZ', 9),
    pytest.param('its own self-UNKWORDZ universe.',
                 7,
                 marks=pytest.mark.xfail(reason='poor handling of -')),
    pytest.param('its own UNKWORDZ-contained universe.',
                 7,
                 marks=pytest.mark.xfail(reason='poor handling of -')),
    pytest.param('Backslashes are UNKWORDZ/cool.',
                 6,
                 marks=pytest.mark.xfail(reason='/ poor handling of /')),
    pytest.param('Backslashes are fun/UNKWORDZ.',
                 6,
                 marks=pytest.mark.xfail(reason='poor handling of /')),
    pytest.param(
        '    ', 0, marks=pytest.mark.xfail(reason='Repeated whitespaces')),
    pytest.param('I like   whitespaces.',
                 4,
                 marks=pytest.mark.xfail(reason='Repeated whitespaces')),
])
def test_spacytokenizer_length(text, length, tokenizer):
    """Test that tokenizer returns strings of the correct length."""
    tokens = tokenizer.tokenize(text)
    assert len(tokens) == length
