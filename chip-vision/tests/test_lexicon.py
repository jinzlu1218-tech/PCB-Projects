from chipvision.text.lexicon import LexiconCorrector, levenshtein


def test_levenshtein():
    assert levenshtein("SANYO", "SANYO") == 0
    assert levenshtein("SLNYO", "SANYO") == 1
    assert levenshtein("2083", "2CB3") == 2


def test_lexicon_corrects_common_ocr_errors():
    corrector = LexiconCorrector(
        ["SANYO", "LC3664NML-10", "JAPAN", "2CB3"],
        max_distance=2,
        min_similarity=0.45,
    )
    assert corrector.correct("LC3864NML -10").text == "LC3664NML-10"
    assert corrector.correct("LC3864NML -10").corrected
    assert corrector.correct("2083").text == "2CB3"
    assert corrector.correct("SLNYO").text == "SANYO"


def test_lexicon_does_not_force_unrelated_text():
    corrector = LexiconCorrector(["SANYO", "LC3664NML-10"], max_distance=1, min_similarity=0.7)
    result = corrector.correct("ABCDEF")
    assert result.text == "ABCDEF"
    assert not result.corrected


def test_lexicon_does_not_merge_two_words():
    corrector = LexiconCorrector(["JAPAN", "2CB3"], max_distance=2, min_similarity=0.45)
    result = corrector.correct("JAPAN 2CB3")
    assert result.text == "JAPAN 2CB3"
    assert not result.corrected
