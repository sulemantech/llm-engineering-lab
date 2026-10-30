"""Stage 1: character-level tokenizer."""


class CharTokenizer:
    """Maps each distinct character in a corpus to an integer id.

    Contract:
      - vocab is the sorted list of unique characters in `text`
      - vocab_size == len(vocab)
      - decode(encode(s)) == s for any s made of known characters
      - encode raises KeyError on an unknown character
    """

    def __init__(self, text: str):
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, s: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
