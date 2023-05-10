import re
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
        # Regexes below fix the punctuation/special characters problem:
        # https://github.com/dianna-ai/dianna/issues/531

        # matches non-whitespace UNKWORDZ non-whitespace
        sentence1 = re.sub(r'(\S)(UNKWORDZ)(\S)', r'\1 \2 \3', sentence)
        # Matches non-whitespace UNKWORDZ whitespace/end-of-string
        sentence2 = re.sub(r'(\S)(UNKWORDZ)(\s|$)', r'\1 \2\3', sentence1)
        # Matches start-of-string/whitespace UNKWORDZ non-whitespace
        sentence3 = re.sub(r'(^|\s)(UNKWORDZ)(\S)', r'\1\2 \3', sentence2)

        tokens = self.spacy_tokenizer(sentence3)

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Paste together with spaces in between."""
        sentence = ' '.join(tokens)
        return sentence
