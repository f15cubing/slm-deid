"""Consistent surrogate replacement. See pipeline/surrogate.py."""

from __future__ import annotations

from pipeline.surrogate import SurrogateMap


def test_same_value_maps_consistently():
    m = SurrogateMap(seed=0)
    a = m.get("NAME", "Sarah")
    b = m.get("NAME", "Sarah")
    assert a == b  # coreference preserved


def test_distinct_values_get_distinct_surrogates():
    m = SurrogateMap(seed=0)
    assert m.get("NAME", "Sarah") != m.get("NAME", "Marcus")


def test_reproducible_across_instances():
    seq = [("NAME", "Sarah"), ("EMAIL", "s@x.edu"), ("NAME", "Marcus")]

    def run():
        m = SurrogateMap(seed=7)
        return [m.get(lbl, v) for lbl, v in seq]

    # Two identically-driven, instance-seeded maps reproduce the WHOLE sequence.
    assert run() == run()


def test_name_shape_preserved():
    m = SurrogateMap(seed=0)
    assert " " not in m.get("NAME", "Sam")  # single-token → given name
    assert " " in m.get("NAME", "Sarah Connor")  # full name → has a space


def test_email_surrogate_looks_like_email():
    assert "@" in SurrogateMap(seed=0).get("EMAIL", "real@school.edu")
