from unittest import TestCase
import numpy as np
import dianna
from tests.test_onnx_runner import generate_data
from tests.utils import load_movie_review_model
from tests.utils import run_model

class ImageKwargs(TestCase):
    """Suite of tests for kwargs to explainers for Images."""

    def test_lime_image_correct_kwargs(self):
        """Test to ensure correct kwargs to lime run without issues."""
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        axis_labels = ('channels', 'y', 'x')
        labels = [1]

        dianna.explain_image(model_filename,
                                input_data,
                                method='LIME',
                                labels=labels,
                                kernel=None,
                                kernel_width=25,
                                verbose=False,
                                feature_selection='auto',
                                random_state=None,
                                axis_labels=axis_labels,
                                preprocess_function=None,
                                top_labels=None,
                                num_features=10,
                                num_samples=10,
                                return_masks=True,
                                positive_only=False,
                                hide_rest=True,
                                )

    def test_lime_image_extra_kwarg(self):
        """Test to ensure extra kwargs to lime raise warnings."""
        model_filename = 'tests/test_data/mnist_model.onnx'
        input_data = generate_data(batch_size=1)[0].astype(np.float32)
        axis_labels = ('channels', 'y', 'x')
        labels = [1]

        with self.assertWarns(Warning):
            dianna.explain_image(model_filename,
                                    input_data,
                                    method='LIME',
                                    labels=labels,
                                    kernel=None,
                                    kernel_width=25,
                                    verbose=False,
                                    feature_selection='auto',
                                    random_state=None,
                                    axis_labels=axis_labels,
                                    preprocess_function=None,
                                    top_labels=None,
                                    num_features=10,
                                    num_samples=10,
                                    return_masks=True,
                                    positive_only=False,
                                    hide_rest=True,
                                    extra_kwarg=None
                                    )

class TextKwargs(TestCase):
    """Suite of tests for kwargs to explainers for Images."""

    def test_rise_text_correct_kwargs(self):
        """Test to ensure correct kwargs to lime run without issues."""
        review = "such a bad movie"

        dianna.explain_text(
                            self.runner,
                            review,
                            tokenizer=self.runner.tokenizer,
                            method='RISE',
                            labels=(1, 0),
                            n_masks=100,
                            feature_res=8,
                            p_keep=0.5,
                            preprocess_function=None,
                            batch_size=100
                            )

    def test_rise_text_extra_kwarg(self):
        """Test to ensure extra kwargs to lime raise warnings."""
        review = "such a bad movie"

        with self.assertWarns(Warning):
            dianna.explain_text(
                                self.runner,
                                review,
                                tokenizer=self.runner.tokenizer,
                                method='RISE',
                                labels=(1, 0),
                                n_masks=100,
                                feature_res=8,
                                p_keep=0.5,
                                preprocess_function=None,
                                batch_size=100,
                                extra_kwarg=None
                                )

    def setUp(self) -> None:
        """Set seed and load runner."""
        np.random.seed(0)
        self.runner = load_movie_review_model()

class TimeseriesKwargs(TestCase):
    """Suite of tests for kwargs to explainers for Images."""

    def test_lime_timeseries_correct_kwargs(self):
        """Test to ensure correct kwargs to lime run without issues."""
        input_data = np.random.random((10, 1))

        dianna.explain_timeseries(
                                    run_model,
                                    input_timeseries=input_data,
                                    method='LIME',
                                    labels=[0,1],
                                    class_names=["summer", "winter"],
                                    kernel_width=25,
                                    verbose=False,
                                    preprocess_function=None,
                                    feature_selection='auto',
                                    num_features=10,
                                    num_samples=10,
                                    num_slices=10,
                                    batch_size=10,
                                    mask_type='mean',
                                    distance_method='cosine',
                                )

    def test_lime_timeseries_extra_kwargs(self):
        """Test to ensure extra kwargs to lime raise warnings."""
        input_data = np.random.random((10, 1))

        with self.assertWarns(Warning):
            dianna.explain_timeseries(
                                        run_model,
                                        input_timeseries=input_data,
                                        method='LIME',
                                        labels=[0,1],
                                        class_names=["summer", "winter"],
                                        kernel_width=25,
                                        verbose=False,
                                        preprocess_function=None,
                                        feature_selection='auto',
                                        num_features=10,
                                        num_samples=10,
                                        num_slices=10,
                                        batch_size=10,
                                        mask_type='mean',
                                        distance_method='cosine',
                                        extra_kwarg=None
                                    )

class TabularKwargs(TestCase):
    """Suite of tests for kwargs to explainers for Images."""

    def test_lime_tabular_correct_kwargs(self):
        """To be implemented."""

    def test_rise_text_extra_kwarg(self):
        """To be implemented."""
