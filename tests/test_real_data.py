"""Day 3 prep — CRAPII loader/converter (synthetic records, no Kaggle needed)."""

import json

from src.common import tags
from src.datagen.quality_gate import gate
from src.datagen.real_data import load_crapii, record_to_example

# "The email address of Michael Jordan is mjordan@nba.com" — a two-token student name + an email.
RECORD = {
    "document": 7,
    "full_text": "The email address of Michael Jordan is mjordan@nba.com",
    "tokens": ["The", "email", "address", "of", "Michael", "Jordan", "is", "mjordan@nba.com"],
    "trailing_whitespace": [True, True, True, True, True, True, True, False],
    "labels": ["O", "O", "O", "O", "B-NAME_STUDENT", "I-NAME_STUDENT", "O", "B-EMAIL"],
}


def test_record_reconstructs_text_and_tags_only_the_name():
    ex = record_to_example(RECORD)
    # text reconstructed exactly from tokens + whitespace
    assert ex.input == RECORD["full_text"]
    # the two-token student name is a single tagged span; the email is NOT tagged
    expected = f"The email address of {tags.wrap('Michael Jordan')} is mjordan@nba.com"
    assert ex.target == expected
    assert [s.text for s in ex.name_spans()] == ["Michael Jordan"]
    assert gate(ex).ok  # passes the quality gate (integrity + schema)


def test_email_is_left_as_untagged_negative():
    ex = record_to_example(RECORD)
    # nothing but the name is tagged -> the email survives untagged in both input and target
    assert "mjordan@nba.com" in tags.unwrap(ex.target)
    assert tags.unwrap(ex.target) == ex.input


def test_multiple_name_mentions_become_separate_spans():
    rec = {
        "document": 8,
        "full_text": "Sarah met Chen yesterday",
        "tokens": ["Sarah", "met", "Chen", "yesterday"],
        "trailing_whitespace": [True, True, True, False],
        "labels": ["B-NAME_STUDENT", "O", "B-NAME_STUDENT", "O"],
    }
    ex = record_to_example(rec)
    assert [s.text for s in ex.name_spans()] == ["Sarah", "Chen"]
    assert ex.target == f"{tags.wrap('Sarah')} met {tags.wrap('Chen')} yesterday"


def test_essay_with_no_names_has_empty_target_tags():
    rec = {
        "document": 9,
        "full_text": "Design thinking helps teams innovate",
        "tokens": ["Design", "thinking", "helps", "teams", "innovate"],
        "trailing_whitespace": [True, True, True, True, False],
        "labels": ["O", "O", "O", "O", "O"],
    }
    ex = record_to_example(rec)
    assert ex.name_spans() == []
    assert ex.target == ex.input


def test_load_crapii_from_file(tmp_path):
    p = tmp_path / "train.json"
    p.write_text(json.dumps([RECORD, RECORD]), encoding="utf-8")
    exs = load_crapii(p, limit=5)
    assert len(exs) == 2
    for ex in exs:
        assert ex.source == "real_crapii"
        assert ex.category == "real"
        ex.validate()


def test_names_only_filter(tmp_path):
    no_name = {
        "document": 10,
        "full_text": "Design thinking",
        "tokens": ["Design", "thinking"],
        "trailing_whitespace": [True, False],
        "labels": ["O", "O"],
    }
    p = tmp_path / "train.json"
    p.write_text(json.dumps([RECORD, no_name]), encoding="utf-8")
    exs = load_crapii(p, names_only=True)
    assert len(exs) == 1 and exs[0].name_spans()
