import string
from abc import ABC
from abc import abstractmethod
from typing import List
import numpy as np


try:
    from torchtext.data import get_tokenizer
except ImportError as err:
    raise ImportError("Failed to import torchtext, please install manually or reinstall dianna with "
                      "text support: `pip install dianna[text]`") from err


class Tokenizer(ABC):
    """
    Abstract base class for tokenizing.

    Has the same interface as (part of) the transformers Tokenizer class.
    """
    def __init__(self, mask_token: str):
        """
        Tokenizer initializer.

        Args:
            mask_token (str): Token used as mask
        """
        self.mask_token = mask_token

    @abstractmethod
    def tokenize(self, sentence: str) -> List[str]:
        """Split sentence into list of tokens."""

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Merge list of tokens back to sentence."""


class SpacyTokenizer(Tokenizer):
    """Spacy tokenizer for natural language."""
    def __init__(self, name: str = 'en_core_web_sm', mask_token: str = "UNKWORDZ"):
        """Spacy tokenizer initalizer.

        Args:
            name: Name of the Spacy tokenizer to use
            mask_token (str): Token used as mask
        """
        super().__init__(mask_token)
        self.spacy_tokenizer = get_tokenizer('spacy', name)

    def tokenize(self, sentence: str) -> List[str]:
        lower_sentence = sentence.lower()
        raw_tokens = self.spacy_tokenizer(lower_sentence)
        # do not consider several special characters in a row as separate tokens
        # special characters in string.punctuation are !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
        # find indices of tokens that are a special character
        indices_in_tokens = np.where([token in string.punctuation for token in raw_tokens])[0]
        indices_in_raw_string = np.where([character in string.punctuation for character in lower_sentence])[0]
        if indices_in_tokens.size == 0:
            # no special characters at all
            return raw_tokens

        # reconstruct list of tokens, combining consecutive special characters
        tokens = []
        steps_in_raw_string = np.diff(indices_in_raw_string)
        special_char_index = 0
        for idx, token in enumerate(raw_tokens):
            if idx not in indices_in_tokens:
                # not a special character
                tokens.append(token)
            elif special_char_index == 0 or steps_in_raw_string[special_char_index - 1] != 1:
                # create new token if this is the first special character, or it does not directly follow another special character
                # in the original input
                tokens.append(token)
                special_char_index += 1
            else:
                # add token to existing one if both are special characters and they are consecutive in the original input
                tokens[-1] = tokens[-1] + token
                special_char_index += 1
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Paste together with spaces in between."""
        sentence = " ".join(tokens)
        return sentence
