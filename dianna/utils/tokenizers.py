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

    MATCH_token_unk_token = re.compile(r'(\S)(UNKWORDZ)(\S)')
    MATCH_token_unk_white = re.compile(r'(\S)(UNKWORDZ)(\s|$)')
    MATCH_white_unk_token = re.compile(r'(^|\s)(UNKWORDZ)(\S)')

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
        sentence = self._fix_whitespace(sentence)

        tokens = self.spacy_tokenizer(sentence)

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Paste together with spaces in between."""
        sentence = ' '.join(tokens)
        return sentence

    def _fix_whitespace(self, sentence: str):
        """Apply fixes for the punctuation/special characters problem.

        For more info, see:
        https://github.com/dianna-ai/dianna/issues/531
        """
        sentence = self.MATCH_token_unk_token.sub(r'\1 \2 \3', sentence)
        sentence = self.MATCH_token_unk_white.sub(r'\1 \2\3', sentence)
        sentence = self.MATCH_white_unk_token.sub(r'\1\2 \3', sentence)
        return sentence
