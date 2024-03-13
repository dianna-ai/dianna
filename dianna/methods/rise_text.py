import numpy as np
from dianna import utils
from dianna.utils.predict import make_predictions
from dianna.utils.rise_utils import normalize


class RISEText:
    """RISE implementation for text based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb."""

    def __init__(self,
                 n_masks=1000,
                 feature_res=8,
                 p_keep=None,
                 preprocess_function=None):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of input data to keep in each mask (Default: auto-tune this value).
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None

    def explain(self,
                model_or_function,
                input_text,
                labels,
                tokenizer=None,
                batch_size=100):
        """Runs the RISE explainer on text.

           The model will be called with masked versions of the input text.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_text (np.ndarray): Text to be explained
            tokenizer: Tokenizer class with tokenize and convert_tokens_to_string methods, and mask_token attribute
            labels (Iterable(int)): Labels to be explained
            batch_size (int): Batch size to use for running the model.

        Returns:
            List of tuples (word, index of word in raw text, importance for target class) for each class.
        """
        if tokenizer is None:
            raise ValueError('Please provide a tokenizer to explain_text.')

        runner = utils.get_function(
            model_or_function, preprocess_function=self.preprocess_function)
        input_tokens = np.asarray(tokenizer.tokenize(input_text))
        num_tokens = len(input_tokens)
        active_p_keep = (self._determine_p_keep(
            input_tokens, runner, tokenizer, self.n_masks, batch_size)
                         if self.p_keep is None else self.p_keep)
        input_shape = (num_tokens, )
        self.masks = self._generate_masks(
            input_shape, active_p_keep,
            self.n_masks)  # Expose masks for to make user inspection possible
        masked_sentences = self._create_masked_sentences(
            input_tokens, self.masks, tokenizer)
        saliencies = self._get_saliencies(runner, masked_sentences, num_tokens,
                                          batch_size, active_p_keep)
        return self._reshape_result(input_tokens, labels, saliencies)

    def _determine_p_keep(self, input_text, runner, tokenizer, n_masks,
                          batch_size):
        """See n_mask default value https://github.com/dianna-ai/dianna/issues/24#issuecomment-1000152233."""
        p_keeps = np.arange(0.1, 1.0, 0.1)
        stds = []
        for p_keep in p_keeps:
            std = self._calculate_mean_class_std(p_keep, runner, input_text,
                                                 tokenizer, n_masks,
                                                 batch_size)
            stds += [std]
        best_i = np.argmax(stds)
        best_p_keep = p_keeps[best_i]
        print(
            f'Rise parameter p_keep was automatically determined at {best_p_keep}'
        )
        return best_p_keep

    def _calculate_mean_class_std(self, p_keep, runner, input_text, tokenizer,
                                  n_masks, batch_size):
        masks = self._generate_masks(input_text.shape, p_keep, n_masks)
        masked = self._create_masked_sentences(input_text, masks, tokenizer)
        predictions = make_predictions(masked, runner, batch_size)
        std_per_class = predictions.std(axis=0)
        return np.max(std_per_class)

    def _generate_masks(self, input_shape, p_keep, n_masks):
        masks = np.random.choice(a=(True, False),
                                 size=(n_masks, ) + input_shape,
                                 p=(p_keep, 1 - p_keep))
        return masks

    def _get_saliencies(self, runner, sentences, num_tokens, batch_size,
                        p_keep):
        self.predictions = make_predictions(sentences, runner, batch_size)
        unnormalized_saliency = self.predictions.T.dot(
            self.masks.reshape(self.n_masks, -1)).reshape(-1, num_tokens)
        return normalize(unnormalized_saliency, self.n_masks, p_keep)

    @staticmethod
    def _reshape_result(input_tokens, labels, saliencies):
        word_indices = list(range(len(input_tokens)))
        return [
            list(zip(input_tokens, word_indices, saliencies[label]))
            for label in labels
        ]

    def _create_masked_sentences(self, tokens, masks, tokenizer):
        tokens_masked_list = [[
            token if keep else tokenizer.mask_token
            for token, keep in zip(tokens, mask)
        ] for mask in masks]
        masked_sentences = [
            tokenizer.convert_tokens_to_string(tokens_masked)
            for tokens_masked in tokens_masked_list
        ]
        return masked_sentences
