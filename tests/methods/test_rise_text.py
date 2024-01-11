"""Unit tests for RISE text."""
from unittest import TestCase
import numpy as np
import dianna.visualization
from dianna.methods.rise_text import RISEText
from dianna.utils import get_function
from tests.utils import assert_explanation_satisfies_expectations
from tests.utils import load_movie_review_model


class RiseOnText(TestCase):
    """Suite of RISE tests for the text case."""

    def test_rise_text(self):
        """Tests exact expected output given a text and model."""
        review = "such a bad movie"
        expected_words = ["such", "a", "bad", "movie"]
        expected_word_indices = [0, 1, 2, 3]
        expected_positive_scores = [0.30, 0.29, 0.04, 0.25]

        positive_explanation = dianna.explain_text(
            self.runner,
            review,
            tokenizer=self.runner.tokenizer,
            labels=(1, 0),
            method="RISE",
            p_keep=0.5,
        )[0]

        assert_explanation_satisfies_expectations(
            positive_explanation,
            expected_positive_scores,
            expected_word_indices,
            expected_words,
        )

    def test_rise_determine_p_keep_for_text(self):
        """Tests exact expected p_keep given a text and model."""
        expected_p_exact_keep = 0.7
        input_text = "such a bad movie"
        runner = get_function(self.runner)
        input_tokens = np.asarray(runner.tokenizer.tokenize(input_text))

        # pylint: disable=protected-access
        p_keep = RISEText()._determine_p_keep(
            input_tokens,
            runner,
            runner.tokenizer,
            n_masks=100,
            batch_size=100,
        )
        assert np.isclose(p_keep, expected_p_exact_keep)

    def setUp(self) -> None:
        """Set seed and load runner."""
        np.random.seed(0)
        self.runner = load_movie_review_model()
