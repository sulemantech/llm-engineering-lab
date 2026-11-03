from llmlab.tokenizer.char import CharTokenizer


if __name__ == "__main__":
    tok = CharTokenizer("Assalam o Alaikum my name is Ali")
    print("vocab:", tok.vocab)
    print("vocab_size:", tok.vocab_size)
    ids = tok.encode("Ali")
    print("encode('Ali'):", ids)
    print("decode:", tok.decode(ids))
