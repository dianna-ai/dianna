from abc import ABC
from abc import abstractmethod
from typing import List


try:
    from torchtext.data import get_tokenizer
except ImportError as err:
    raise ImportError(
        'Failed to import torchtext, please install manually or reinstall dianna with '
        'text support: `pip install dianna[text]`') from err


class Tokenizer(ABC):
    """Abstract base class for tokenizing.

    Has the same interface as (part of) the transformers Tokenizer class.
    """

    def __init__(self, mask_token: str):
        """Tokenizer initializer.

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


from itertools import tee


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


import re


retokenizer = re.compile(r'(\w+|\S)')


class SpacyTokenizer(Tokenizer):
    """Spacy tokenizer for natural language."""

    def __init__(self,
                 name: str = 'en_core_web_sm',
                 mask_token: str = 'UNKWORDZ'):
        """Spacy tokenizer initalizer.

        Args:
            name: Name of the Spacy tokenizer to use
            mask_token (str): Token used as mask
        """
        super().__init__(mask_token)
        self.spacy_tokenizer = get_tokenizer('spacy', name)

    def tokenize(self, sentence: str) -> List[str]:
        """Tokenize sentence."""
        sentence1 = re.sub(r'(\w)(UNKWORDZ)(\s|$)', r'\1 \2\3', sentence)
        sentence2 = re.sub(r'(^|\s)(UNKWORDZ)(\w)', r'\1\2 \3', sentence1)

        tokens = self.spacy_tokenizer(sentence2)

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Paste together with spaces in between."""
        sentence = ' '.join(tokens)
        return sentence
