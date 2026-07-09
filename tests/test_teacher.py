"""Day 2, spec S2.8 — teacher distillation parse + second-pass verify (mocked).

Day 4 additions (TASK 1): matched minimal-pair generation + category hints that no longer seed
the quarantined eval tokens.
"""

import re

from src.common import tags
from src.datagen.quality_gate import gate
from src.datagen.teacher import _CATEGORY_HINT, TeacherGenerator, parse_tagged
from src.datagen.vocab import BLOCKLIST


def test_parse_tagged_derives_input_target_spans():
    text = f"We used the Newton method, but {tags.wrap('Newton')} himself would frown."
    raw, target, spans = parse_tagged(text)
    assert raw == "We used the Newton method, but Newton himself would frown."
    assert target == text
    assert [s.text for s in spans] == ["Newton"]
    assert raw[spans[0].start : spans[0].end] == "Newton"


def test_parse_strips_code_fences():
    text = f"```\n{tags.wrap('Ada')} coded.\n```"
    raw, target, _ = parse_tagged(text)
    assert target == f"{tags.wrap('Ada')} coded."
    assert raw == "Ada coded."


def test_generate_produces_valid_example():
    canned = f"{tags.wrap('Grace')} lent me her notes, but she showed grace under pressure."
    gen = TeacherGenerator(gen=lambda s, u: canned)
    ex = gen.generate("person_vs_common", register="dialogue", id_="t1")
    assert gate(ex).ok
    assert [s.text for s in ex.name_spans()] == ["Grace"]


def test_second_pass_agreement_passes_gate():
    canned = f"{tags.wrap('Grace')} lent me her notes, but she showed grace under pressure."
    # verifier tags identically -> agreement
    gen = TeacherGenerator(gen=lambda s, u: canned, verify=lambda s, u: canned)
    ex = gen.generate("person_vs_common", id_="t2")
    vt = gen.verify_tagging(ex.input)
    assert gate(ex, verifier_target=vt).ok


def test_category_hints_no_longer_seed_eval_tokens():
    # Root problem 1: the old hints listed the exact eval tokens (Newton/Chelsea/Grace/…).
    blob = " ".join(_CATEGORY_HINT.values()).lower()
    hits = sorted(b for b in BLOCKLIST if re.search(rf"\b{re.escape(b)}\b", blob))
    assert not hits, f"category hints still seed eval tokens: {hits}"


def test_generate_minimal_pair_person_vs_place():
    # A matched pair for one surface token: person (tagged) + place (untagged).
    def gen(system, user):
        if "SENSE=person" in user:
            return f"{tags.wrap('Austin')} explained the recursion clearly after class."
        return "We drove through Austin on the way to the coast last weekend."

    t = TeacherGenerator(gen=gen)
    person, nonperson = t.generate_pair("person_vs_place", person_token="Austin")

    assert person.ambiguous_token == "Austin" and nonperson.ambiguous_token == "Austin"
    assert person.category == nonperson.category == "person_vs_place"
    assert [s.text for s in person.name_spans()] == ["Austin"]  # person -> tagged
    assert nonperson.name_spans() == []  # place -> untagged
    assert "austin" in person.input.lower() and "austin" in nonperson.input.lower()
    assert gate(person).ok and gate(nonperson).ok


def test_generate_minimal_pair_possessive_person_vs_eponymous():
    # possessive pair: a person's possessive vs an eponymous possessive (distinct tokens).
    def gen(system, user):
        if "SENSE=person" in user:
            return f"{tags.wrap('Joy')}'s essay improved after peer review."
        return "Joule's law relates power to current and resistance."

    t = TeacherGenerator(gen=gen)
    person, nonperson = t.generate_pair("possessive", person_token="Joy", nonperson_token="Joule")
    assert person.ambiguous_token == "Joy" and nonperson.ambiguous_token == "Joule"
    assert [s.text for s in person.name_spans()] == ["Joy"]
    assert nonperson.name_spans() == []
    assert gate(person).ok and gate(nonperson).ok


def test_second_pass_disagreement_dropped():
    raw_tagged = f"{tags.wrap('Grace')} lent me her notes, but she showed grace under pressure."

    # verifier wrongly tags the concept "grace" too -> disagreement -> gate drops it
    def verify(s, u):
        return (
            f"{tags.wrap('Grace')} lent me her notes, but she showed "
            f"{tags.wrap('grace')} under pressure."
        )

    gen = TeacherGenerator(gen=lambda s, u: raw_tagged, verify=verify)
    ex = gen.generate("person_vs_common", id_="t3")
    vt = gen.verify_tagging(ex.input)
    r = gate(ex, verifier_target=vt)
    assert not r.ok and r.reason == "verifier_disagreement"
