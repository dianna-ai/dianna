from abc import ABC, abstractmethod
from typing import List
import string


class Tokenizer(ABC):
    """
    Abstract base class for tokenizing.
    Has the same interface as (part of) the transformers Tokenizer class.
    """
    def __init__(self, mask_token: str ):
        self.mask_token = mask_token
    
    @abstractmethod
    def tokenize(self, sentence: str) -> List[str]:
        """
        Split sentence into list of tokens.
        """
        pass
    
    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Merges list of tokens back to sentence.
        """
        pass


class WordBasedTokenizer(Tokenizer):
    def __init__(self, mask_token: str = "UNKWORDZ"):
        super().__init__(mask_token)
    
    def tokenize(self, sentence: str) -> List[str]:
        """Strip punctuation and split at spaces."""
        sentence_nopunc = sentence.translate(str.maketrans('', '', string.punctuation))
        tokens = sentence_nopunc.split(' ')
        return tokens
         
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Paste together with spaces in between."""
        sentence = " ".join(tokens)
        return sentence