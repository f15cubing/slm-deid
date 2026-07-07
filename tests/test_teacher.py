"""Day 2, spec S2.8 — teacher distillation parse + second-pass verify (mocked)."""

from src.common import tags
from src.datagen.quality_gate import gate
from src.datagen.teacher import TeacherGenerator, parse_tagged


def test_parse_tagged_derives_input_target_spans():
    text = f"We used the Newton method, but {tags.wrap('Newton')} himself would frown."
    raw, target, spans = parse_tagged(text)
    assert raw == "We used the Newton method, but Newton himself would frown."
    assert target == text
    assert [s.text for s in spans] == ["Newton"]
    assert raw[spans[0].start:spans[0].end] == "Newton"


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
