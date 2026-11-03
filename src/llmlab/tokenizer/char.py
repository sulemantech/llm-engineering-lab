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
        self.vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
     

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
         return "".join(self.itos[i] for i in ids)


# if __name__ == "__main__":
#     tok = CharTokenizer("Assalam o Alaikum my name is Ali")
#     print("vocab:", tok.vocab)
#     print("vocab_size:", tok.vocab_size)
#     ids = tok.encode("Ali")
#     print("encode('Ali'):", ids)
#     print("decode:", tok.decode(ids))
