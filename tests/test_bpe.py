from astralm.bpe import BytePairTokenizer


def test_bpe_unicode_round_trip() -> None:
    text = "星河模型 learns patterns. 星河模型 learns patterns."
    tokenizer = BytePairTokenizer.train(text, vocab_size=280)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.compression_ratio(text) > 1.0


def test_bpe_training_is_deterministic() -> None:
    text = "abc abc abc xyz xyz"
    assert BytePairTokenizer.train(text, 270).merges == BytePairTokenizer.train(text, 270).merges


def test_bpe_save_and_load(tmp_path) -> None:
    text = "cache cache cache bridge bridge"
    tokenizer = BytePairTokenizer.train(text, 270)
    loaded = BytePairTokenizer.load(tokenizer.save(tmp_path / "tokenizer.json"))
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
