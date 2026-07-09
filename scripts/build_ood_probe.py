"""Build the out-of-distribution (OOD) generalization probe — a quarantined hard-cases split
whose ambiguous surfaces are disjoint from BOTH the standard eval set and the training vocab banks.

Motivation
----------
The standard ``eval/hardcases`` set and the training data (via ``src/datagen/vocab.py``) share the
same *categories* (person_vs_eponym, person_vs_place, …) but the Day-4 rebalance already guarantees
the training banks share **no token** with the eval set. This probe goes one step further: every
ambiguous surface here is absent from the eval BLOCKLIST *and* from the training banks
(PLACES / COMMON_WORDS / EPONYMS). So these surfaces are genuinely novel to the tuned model — it saw
neither them in training nor in the standard eval. If judgment still holds here, it generalized the
*rule* rather than memorizing vocabulary.

The set is quarantined (``quarantine=true``, ``source="handbuilt"``) so the eval-leakage guard
(``tests/test_no_eval_leakage.py``) and the bank-disjointness guard (``tests/test_vocab.py``) both
cover it automatically. Regenerate with::

    PYTHONPATH=. python scripts/build_ood_probe.py

Run the probe (Colab/CUDA or Mac/MPS) with::

    python -m src.eval.run --split eval/ood_probe --compare base <adapter>
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.common import tags
from src.datagen.vocab import BLOCKLIST, COMMON_WORDS, EPONYMS, PLACES

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "eval" / "ood_probe" / "ood_hardcases.jsonl"

# (id, input, name|None, category, register, ambiguous_token, paraphrase_group)
CASES = [
    # ---- person_vs_eponym (fresh eponyms: Boyle/Kepler/Morse/Diesel/Geiger) ----
    (
        "ood-boyle-person",
        "Boyle stayed after lab to help me recalibrate the sensor.",
        "Boyle",
        "person_vs_eponym",
        "essay",
        "boyle",
        "boyle",
    ),
    (
        "ood-boyle-law",
        "Boyle's law relates the pressure and volume of a gas.",
        None,
        "person_vs_eponym",
        "essay",
        "boyle",
        "boyle",
    ),
    (
        "ood-kepler-person",
        "Kepler emailed the seminar slides an hour before class.",
        "Kepler",
        "person_vs_eponym",
        "essay",
        "kepler",
        "kepler",
    ),
    (
        "ood-kepler-laws",
        "We derived Kepler's laws of planetary motion for homework.",
        None,
        "person_vs_eponym",
        "essay",
        "kepler",
        "kepler",
    ),
    (
        "ood-diesel-engine",
        "The bus still runs on an old diesel engine.",
        None,
        "person_vs_eponym",
        "essay",
        "diesel",
        None,
    ),
    (
        "ood-morse-code",
        "The scouts learned to signal SOS in Morse code.",
        None,
        "person_vs_eponym",
        "essay",
        "morse",
        "morse",
    ),
    (
        "ood-morse-person",
        "Morse graded our problem sets over the weekend.",
        "Morse",
        "person_vs_eponym",
        "dialogue",
        "morse",
        "morse",
    ),
    (
        "ood-geiger-counter",
        "The technician swept the room with a Geiger counter.",
        None,
        "person_vs_eponym",
        "essay",
        "geiger",
        None,
    ),
    # ---- person_vs_place (fresh place-names: Jackson/Victoria/Geneva/Berlin) ----
    (
        "ood-jackson-city",
        "We drove through Jackson on the way to the canyon.",
        None,
        "person_vs_place",
        "essay",
        "jackson",
        "jackson",
    ),
    (
        "ood-jackson-person",
        "Jackson walked me through the proof twice.",
        "Jackson",
        "person_vs_place",
        "dialogue",
        "jackson",
        "jackson",
    ),
    (
        "ood-victoria-place",
        "The harbor in Victoria was crowded with ferries.",
        None,
        "person_vs_place",
        "essay",
        "victoria",
        "victoria",
    ),
    (
        "ood-victoria-person",
        "Victoria offered to co-author the section with me.",
        "Victoria",
        "person_vs_place",
        "essay",
        "victoria",
        "victoria",
    ),
    (
        "ood-geneva-place",
        "The treaty was signed in Geneva after months of talks.",
        None,
        "person_vs_place",
        "essay",
        "geneva",
        "geneva",
    ),
    (
        "ood-geneva-person",
        "thanks Geneva, that hint unblocked my whole solution",
        "Geneva",
        "person_vs_place",
        "dialogue",
        "geneva",
        "geneva",
    ),
    (
        "ood-berlin-place",
        "The wall came down in Berlin in 1989.",
        None,
        "person_vs_place",
        "essay",
        "berlin",
        None,
    ),
    # ---- person_vs_common (fresh common-word names: Summer/Will/Carol/Grant) ----
    (
        "ood-summer-season",
        "Summer felt endless once the exams were over.",
        None,
        "person_vs_common",
        "essay",
        "summer",
        "summer",
    ),
    (
        "ood-summer-person",
        "Summer lent me her graphing calculator for the final.",
        "Summer",
        "person_vs_common",
        "dialogue",
        "summer",
        "summer",
    ),
    (
        "ood-will-modal",
        "I will finish the lab writeup before midnight.",
        None,
        "person_vs_common",
        "essay",
        "will",
        "will",
    ),
    (
        "ood-will-person",
        "Will rewrote the introduction and it reads much better now.",
        "Will",
        "person_vs_common",
        "essay",
        "will",
        "will",
    ),
    (
        "ood-carol-song",
        "The choir practiced a carol for the winter assembly.",
        None,
        "person_vs_common",
        "essay",
        "carol",
        "carol",
    ),
    (
        "ood-carol-person",
        "Carol asked the sharpest question in the whole review session.",
        "Carol",
        "person_vs_common",
        "dialogue",
        "carol",
        "carol",
    ),
    (
        "ood-grant-funding",
        "The research grant covered our travel to the conference.",
        None,
        "person_vs_common",
        "essay",
        "grant",
        "grant",
    ),
    (
        "ood-grant-person",
        "Grant proofread my intro and caught two logic gaps.",
        "Grant",
        "person_vs_common",
        "dialogue",
        "grant",
        "grant",
    ),
    # ---- possessive (person possessive positive; eponymous possessive negative) ----
    (
        "ood-poss-nadia",
        "Nadia's presentation set the bar for the rest of us.",
        "Nadia",
        "possessive",
        "essay",
        "nadia",
        None,
    ),
    (
        "ood-poss-halley",
        "Halley's comet will not be visible again for decades.",
        None,
        "possessive",
        "essay",
        "halley",
        None,
    ),
    # ---- first_name_only (fresh, non-Western surfaces buried mid-sentence) ----
    (
        "ood-first-okafor",
        "When Okafor raised her hand, the whole room turned.",
        "Okafor",
        "first_name_only",
        "essay",
        "okafor",
        None,
    ),
    (
        "ood-first-tobias",
        "ok Tobias i'll push the fix in a minute",
        "Tobias",
        "first_name_only",
        "dialogue",
        "tobias",
        None,
    ),
    # ---- third_party (someone other than the writer) ----
    (
        "ood-third-nakamura",
        "My advisor, Dr. Nakamura, moved the defense to Thursday.",
        "Nakamura",
        "third_party",
        "essay",
        "nakamura",
        None,
    ),
    (
        "ood-third-kofi",
        "my roommate Kofi already submitted the group report",
        "Kofi",
        "third_party",
        "dialogue",
        "kofi",
        None,
    ),
    # ---- negative_trap (titles / brands / orgs / sentence-initial subjects) ----
    (
        "ood-neg-course",
        "Organic Chemistry met every Thursday in the annex.",
        None,
        "negative_trap",
        "essay",
        "chemistry",
        None,
    ),
    (
        "ood-neg-brand",
        "i typed the entire essay on my Chromebook at the library",
        None,
        "negative_trap",
        "dialogue",
        "chromebook",
        None,
    ),
    (
        "ood-neg-org",
        "The United Nations sent observers to monitor the election.",
        None,
        "negative_trap",
        "essay",
        "united nations",
        None,
    ),
    (
        "ood-neg-subject",
        "Calculus turned out to be my favorite class this term.",
        None,
        "negative_trap",
        "essay",
        "calculus",
        None,
    ),
    # ---- easy (unambiguous names, should always be tagged) ----
    (
        "ood-easy-okonkwo",
        "Dear Dr. Okonkwo, thank you for the detailed feedback.",
        "Okonkwo",
        "easy",
        "essay",
        "okonkwo",
        None,
    ),
]

# Multi-name easy cases (two names / full-name span), handled by ``make_multi``.
MULTI = [
    (
        "ood-easy-two",
        "Lena and Marcus split the presentation between them.",
        ["Lena", "Marcus"],
        "easy",
        "dialogue",
        "lena+marcus",
        None,
    ),
    (
        "ood-easy-signed",
        "Sincerely, Wei Zhang",
        ["Wei Zhang"],
        "easy",
        "essay",
        "wei zhang",
        None,
    ),
]


def make(id_, text, name, category, register, tok, group):
    if name is None:
        spans = []
        target = text
    else:
        start = text.index(name)
        end = start + len(name)
        spans = [{"start": start, "end": end, "text": name, "is_name": True}]
        target = text[:start] + tags.wrap(name) + text[end:]
    return {
        "id": id_,
        "input": text,
        "target": target,
        "register": register,
        "category": category,
        "spans": spans,
        "source": "handbuilt",
        "paraphrase_group": group,
        "quarantine": True,
        "ambiguous_token": tok,
    }


def make_multi(id_, text, names, category, register, tok, group):
    spans, target, cursor = [], "", 0
    idxs = []
    search_from = 0
    for nm in names:
        s = text.index(nm, search_from)
        idxs.append((s, s + len(nm), nm))
        search_from = s + len(nm)
    for s, e, nm in idxs:
        spans.append({"start": s, "end": e, "text": nm, "is_name": True})
    for s, e, nm in idxs:
        target += text[cursor:s] + tags.wrap(nm)
        cursor = e
    target += text[cursor:]
    return {
        "id": id_,
        "input": text,
        "target": target,
        "register": register,
        "category": category,
        "spans": spans,
        "source": "handbuilt",
        "paraphrase_group": group,
        "quarantine": True,
        "ambiguous_token": tok,
    }


def build() -> list[dict]:
    rows = [make(*c) for c in CASES]
    rows += [make_multi(*c) for c in MULTI]

    # OOD guard: no ambiguous surface may collide with the training banks or the eval BLOCKLIST.
    train_surfaces = {t.lower() for t in (*PLACES, *COMMON_WORDS, *EPONYMS)}
    collisions = []
    for r in rows:
        for word in re.findall(r"[a-z]+", r["ambiguous_token"]):
            if word in train_surfaces or word in BLOCKLIST:
                collisions.append((r["id"], word))
    if collisions:
        raise SystemExit(f"OOD guard FAILED — surface overlaps training/eval: {collisions}")
    return rows


def main() -> None:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    pos = sum(1 for r in rows if r["spans"])
    cats: dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"wrote {len(rows)} OOD cases ({pos} positive / {len(rows) - pos} negative) -> {OUT}")
    print("by category:", dict(sorted(cats.items())))
    print("OOD guard: PASSED (no surface overlaps training banks or eval BLOCKLIST)")


if __name__ == "__main__":
    main()
