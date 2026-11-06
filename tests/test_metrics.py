import json
from pathlib import Path

from eval.metrics import cer_counts, corpus_scores, edit_distance, normalize_devanagari
from src.pipeline import HinglishAssistant


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


def test_normalization_collapses_variants():
    assert normalize_devanagari("हूँ।") == normalize_devanagari("हूं")
    assert normalize_devanagari("ज़रूर") == normalize_devanagari("जरूर")


def test_perfect_and_imperfect_scores():
    s = corpus_scores(["मेरा नाम राहुल है"], ["मेरा नाम राहुल है"])
    assert s["cer"] == 0 and s["word_accuracy"] == 1 and s["exact_match"] == 1
    s = corpus_scores(["मेरा नाम राहुल"], ["मेरा नाम राहुल है"])
    assert 0 < s["cer"] < 1 and s["word_accuracy"] < 1


def test_cer_ignores_spaces():
    assert cer_counts("मेरा नाम", "मेराा नाम")[0] == 1


def test_case_file_shape():
    d = json.loads((Path(__file__).parents[1] / "eval" / "test_cases.json").read_text(encoding="utf-8"))
    assert len(d["transliteration"]) == 40 and len(d["translation"]) == 10
    ids = [c["id"] for c in d["transliteration"] + d["translation"]]
    assert len(set(ids)) == 50


def test_pipeline_with_mock_backend():
    r = HinglishAssistant("mock").run("mera naam rahul hai", translate=True, polish=True)
    assert r.native and r.polished and r.english and not r.warnings
    assert HinglishAssistant("mock").run("   ").warnings
