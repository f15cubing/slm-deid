"""Generate a HELD-OUT-NAMES generalization test set (quarantined).

Motivation
----------
The quarantined ``eval/hardcases`` set proves the model tags the *specific* ambiguous surfaces
the plan enumerates (Newton / Chelsea / Grace / …). But those surfaces are also the ones the
data-gen ``BLOCKLIST`` reserves, so hardcases mostly answers "does the judgment hold on the
handful of canonical traps?". It does not answer the sharper question the user asked:

    Does the name judgment generalize to names the model has NEVER seen — neither in training
    nor in the existing eval?

This script builds exactly that probe. Every ambiguous surface here is verified **disjoint** from
(a) the training/val/co-occurrence text and its ``ambiguous_token`` labels, (b) the data-gen vocab
bank + BLOCKLIST, and (c) the existing ``eval/hardcases`` vocabulary. The novel variable is the
**name**: the context frames are the same *style* as training (educational essay / chat) and some
rows are close paraphrases of ``src/datagen/author.py`` templates with only the name swapped. So
this isolates "does the judgment transfer to an unseen name?", not "unseen phrasing" — see the
scope caveat in ``docs/heldout-names-testset.md``.

The set covers the plan's hard categories — person-vs-place, person-vs-common, person-vs-eponym,
possessive (person vs eponymous), first-name-only, third-party, negative-trap — with 2–3
paraphrases per (name, sense) so consistency is measurable.

Output: ``eval/heldout_names/heldout_names.jsonl`` (quarantine=True). This file is NEVER fed to
data-gen or training; ``tests/test_heldout_names_disjoint.py`` pins the disjointness invariant.

Run: ``python scripts/gen_heldout_names_testset.py`` (pure/offline; no model, no network).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.schema import Example, Span, read_jsonl, write_jsonl  # noqa: E402
from src.datagen import vocab  # noqa: E402

OUT = ROOT / "eval" / "heldout_names" / "heldout_names.jsonl"

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


# --- forbidden vocabulary (everything the model may already have "seen") -----------------------
def forbidden_vocab() -> set[str]:
    """Union of every lowercased word the model could have seen as training/bank/eval signal."""
    forbidden: set[str] = set()

    # (a) training/val/co-occurrence: passage + target text AND the ambiguous_token labels.
    for rel in ("data/splits/train.jsonl", "data/splits/val.jsonl", "data/cooccur/cooccur.jsonl"):
        p = ROOT / rel
        if not p.exists():
            continue
        for ex in read_jsonl(p):
            forbidden |= _words(ex.input)
            forbidden |= _words(ex.target)
            if ex.ambiguous_token:
                forbidden |= _words(ex.ambiguous_token)

    # (b) the data-gen vocab bank + BLOCKLIST (the teacher's whole name pool).
    for tok in vocab.all_bank_tokens():
        forbidden |= _words(tok)
    for tok in vocab.BLOCKLIST:
        forbidden |= _words(tok)

    # (c) the existing quarantined eval vocabulary (so this is a fresh, non-overlapping probe).
    forbidden |= vocab.eval_vocab(str(ROOT / "eval" / "hardcases"))

    return forbidden


# --- curated minimal-pair specs (fresh names; training-style context frames) ------------------
# Each spec: (token, [person paraphrases], [non-person paraphrases]). Sentences mention the token
# with normal capitalization; the tagger tags EVERY word-boundary occurrence of the token in the
# person sentences and nothing in the non-person ones. Names are picked to be absent from
# forbidden_vocab() — any that slip through are dropped with a warning (never silently kept).

# person_vs_place: a place name used as a person vs as a place.
PLACE_SPECS = [
    (
        "Jackson",
        [
            "Jackson stayed behind to help me reformat the bibliography.",
            "our group leader Jackson split the workload evenly before the deadline",
        ],
        [
            "For the geography unit we charted how Jackson expanded after the flood.",
            "the museum exhibit about Jackson traced its founding and early trade",
        ],
    ),
    (
        "Victoria",
        [
            "Victoria walked me through the derivation twice until it clicked.",
            "thanks to Victoria i finally understood the second reading",
        ],
        [
            "The report compared rainfall in Victoria to the neighboring region.",
            "we looked at ferry routes around Victoria for the coastal-cities project",
        ],
    ),
    (
        "Aurora",
        [
            "Aurora reorganized our slides an hour before the presentation.",
            "Aurora and I rehearsed the argument section together at lunch.",
        ],
        [
            "The article described how Aurora grew around the old rail depot.",
            "did you find a population source for Aurora yet for the map?",
        ],
    ),
    (
        "Marion",
        [
            "Marion caught the error in my citation format right away.",
            "i asked Marion to peer-review the intro paragraph last night",
        ],
        [
            "We mapped the towns near Marion for the county history assignment.",
            "the documentary about Marion covered its rivers and farmland",
        ],
    ),
]

# person_vs_common: a common word that is also a given name.
COMMON_SPECS = [
    (
        "Summer",
        [
            "Summer rewrote the conclusion so it actually answered the prompt.",
            "honestly Summer explained the whole proof better than the textbook did",
        ],
        [
            "My essay opens with an image of summer fading into autumn.",
            "the reading had this long passage about summer on the coast",
        ],
    ),
    (
        "Art",
        [
            "Art volunteered to present our findings to the rest of the class.",
            "i sat next to Art during the entire review session",
        ],
        [
            "Our art teacher pushed the portfolio deadline back a week.",
            "the class debated whether graffiti counts as art for the essay",
        ],
    ),
    (
        "Willow",
        [
            "Willow double-checked my calculations before I submitted the lab.",
            "Willow reminded the group to cite the second source.",
        ],
        [
            "The poem lingers on a willow bending over the still pond.",
            "we sketched a willow by the river for the botany worksheet",
        ],
    ),
    (
        "Mercy",
        [
            "Mercy stayed late to quiz me on the vocabulary list.",
            "can you ask Mercy to send over her notes from Tuesday?",
        ],
        [
            "The essay argued that mercy should temper the justice system.",
            "the play keeps returning to the theme of mercy in its final act",
        ],
    ),
    (
        "Meadow",
        [
            "Meadow organized our sources into a shared document for the group.",
            "i think Meadow already outlined the whole presentation",
        ],
        [
            "A wide meadow stretched behind the school in the closing scene.",
            "the poem sets its final image in a meadow at dusk",
        ],
    ),
    (
        "Poppy",
        [
            "Poppy volunteered to record the group's data during the experiment.",
            "thanks to Poppy our slides were finished a day early",
        ],
        [
            "A single poppy in the field became the essay's central symbol.",
            "we pressed a poppy between the pages for the botany project",
        ],
    ),
    (
        "Laurel",
        [
            "Laurel reworked the thesis statement so it matched the evidence.",
            "can you check whether Laurel finished the annotated bibliography?",
        ],
        [
            "The victor received a crown of laurel in the myth we read.",
            "a sprig of laurel appears twice in the second stanza",
        ],
    ),
    (
        "Juniper",
        [
            "Juniper walked the group through the lab safety rules again.",
            "honestly Juniper kept our whole project on schedule",
        ],
        [
            "A gnarled juniper marked the trailhead in the field notes.",
            "the guide pointed out a juniper growing between the rocks",
        ],
    ),
]

# person_vs_eponym: a surname that doubles as a unit/law/method.
EPONYM_SPECS = [
    (
        "Kepler",
        [
            "Kepler stayed after class to help me set up the telescope demo.",
            "our lab partner Kepler recalibrated the sensor before the trial",
        ],
        [
            "We used Kepler's third law to relate the orbital periods.",
            "the derivation leaned on the Kepler equation from chapter nine",
        ],
    ),
    (
        "Snell",
        [
            "Snell explained refraction to me until the diagram made sense.",
            "i think Snell already finished the optics problem set",
        ],
        [
            "We applied the Snell relation to find the angle of refraction.",
            "the Snell law showed up on every problem in the review packet",
        ],
    ),
    (
        "Bragg",
        [
            "Bragg rebuilt the lattice model after the first one collapsed.",
            "Bragg walked our table through the diffraction lab step by step.",
        ],
        [
            "The proof relied on the Bragg condition for constructive interference.",
            "we measured the spacing using the Bragg equation from the handout",
        ],
    ),
    (
        "Nernst",
        [
            "Nernst offered to quiz us on the electrochemistry unit.",
            "thanks to Nernst i finally understood half-cell potentials",
        ],
        [
            "The Nernst equation let us correct the potential for concentration.",
            "our chemistry unit introduced the Nernst relation midway through",
        ],
    ),
]

# possessive: person's possessive (tag the name) vs eponymous possessive (tag nothing).
POSSESSIVE_SPECS = [
    (
        "Willow",
        [
            "Willow's essay improved a lot after the peer-editing round.",
            "I borrowed Willow's notes to catch up on the lecture.",
        ],
        [],  # non-person possessive handled by the eponym half below
    ),
    (
        "Summer",
        [
            "Summer's revision of the introduction was the clearest in the class.",
            "we followed Summer's outline to structure the group report",
        ],
        [],
    ),
]
# eponymous possessives (nothing tagged) reuse fresh eponym surnames.
EPONYM_POSSESSIVE = [
    (
        "Kepler",
        [
            "Kepler's law describes the orbit's shape precisely.",
            "the chapter derived Kepler's equation step by step",
        ],
    ),
    (
        "Bragg",
        [
            "Bragg's law relates the spacing to the diffraction angle.",
            "we applied Bragg's condition to the mineral sample",
        ],
    ),
]

# first_name_only: a buried first name in dialogue (tag it). Uses fresh names.
FIRST_NAME_SPECS = [
    ("Marion", ["thanks, Marion — that clears up the whole assignment"]),
    ("Art", ["good catch, Art, i totally missed that citation"]),
    ("Aurora", ["appreciate it Aurora, i'll fix the intro tonight"]),
    ("Snell", ["yeah Snell, the optics lab is due friday right?"]),
]

# third_party: the author names a friend/teacher/parent (still a personal name -> tag).
THIRD_PARTY_SPECS = [
    ("Victoria", ["My friend Victoria helped me study for the chemistry final."]),
    ("Jackson", ["My older brother Jackson proofread the essay before I submitted it."]),
    ("Mercy", ["Our tutor Mercy suggested I restructure the second paragraph."]),
]

# negative_trap: NO personal name present -> nothing tagged. Sentence-initial capitals, brands,
# course titles, capitalized concepts. These share no ambiguous surface with training either.
NEGATIVE_TRAPS = [
    "Reading the assigned chapter twice made the argument much clearer.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Introduction to Statistics meets on Tuesday and Thursday mornings.",
    "Chromebook batteries drained fast during the all-day exam.",
    "Whenever the lab ran long, we cleaned the glassware before leaving.",
    "Advanced Placement Biology covered cellular respiration this week.",
]


def _tag_all_occurrences(text: str, token: str) -> tuple[str, list[Span]]:
    """Wrap every word-boundary occurrence of ``token`` in ``text`` and return (target, spans)."""
    pat = re.compile(rf"\b{re.escape(token)}\b")
    matches = list(pat.finditer(text))
    if not matches:
        raise ValueError(f"token {token!r} not found as a whole word in {text!r}")
    spans = [Span(m.start(), m.end(), m.group(), True) for m in matches]
    # Build target by inserting tags from right to left so earlier offsets stay valid.
    target = text
    for m in reversed(matches):
        target = target[: m.start()] + "⟨NAME⟩" + m.group() + "⟨/NAME⟩" + target[m.end() :]
    return target, spans


def _person(idx: int, category: str, token: str, sentence: str) -> Example:
    register = "dialogue" if sentence[:1].islower() else "essay"
    target, spans = _tag_all_occurrences(sentence, token)
    return Example(
        id=f"heldout-{category}-{token}-person-{idx}",
        input=sentence,
        target=target,
        register=register,
        category=category,
        spans=spans,
        source="handbuilt",
        paraphrase_group=f"{category}:{token}:person",
        quarantine=True,
        ambiguous_token=token,
    )


def _nonperson(idx: int, category: str, token: str | None, sentence: str) -> Example:
    register = "dialogue" if sentence[:1].islower() else "essay"
    group = f"{category}:{token}:nonperson" if token else None
    return Example(
        id=f"heldout-{category}-{token or 'trap'}-nonperson-{idx}",
        input=sentence,
        target=sentence,  # nothing tagged -> byte-identical
        register=register,
        category=category,
        spans=[],
        source="handbuilt",
        paraphrase_group=group,
        quarantine=True,
        ambiguous_token=token,
    )


def build() -> list[Example]:
    forbidden = forbidden_vocab()
    kept: list[Example] = []
    dropped: list[str] = []

    def token_ok(token: str) -> bool:
        overlap = _words(token) & forbidden
        if overlap:
            dropped.append(f"{token} (overlaps {sorted(overlap)})")
            return False
        return True

    # two-sided specs (person + non-person)
    two_sided = [
        ("person_vs_place", PLACE_SPECS),
        ("person_vs_common", COMMON_SPECS),
        ("person_vs_eponym", EPONYM_SPECS),
    ]
    for category, specs in two_sided:
        for token, persons, nons in specs:
            if not token_ok(token):
                continue
            for i, s in enumerate(persons):
                kept.append(_person(i, category, token, s))
            for i, s in enumerate(nons):
                kept.append(_nonperson(i, category, token, s))

    # possessive: person half (tag name) + eponymous half (tag nothing)
    for token, persons, _ in POSSESSIVE_SPECS:
        if not token_ok(token):
            continue
        for i, s in enumerate(persons):
            kept.append(_person(i, "possessive", token, s))
    for token, nons in EPONYM_POSSESSIVE:
        if not token_ok(token):
            continue
        for i, s in enumerate(nons):
            kept.append(_nonperson(i, "possessive", token, s))

    # first_name_only (person)
    for token, sents in FIRST_NAME_SPECS:
        if not token_ok(token):
            continue
        for i, s in enumerate(sents):
            kept.append(_person(i, "first_name_only", token, s))

    # third_party (person)
    for token, sents in THIRD_PARTY_SPECS:
        if not token_ok(token):
            continue
        for i, s in enumerate(sents):
            kept.append(_person(i, "third_party", token, s))

    # negative_trap (no name; no paraphrase group)
    for i, s in enumerate(NEGATIVE_TRAPS):
        kept.append(_nonperson(i, "negative_trap", None, s))

    if dropped:
        print("DROPPED (overlapped forbidden vocab):")
        for d in dropped:
            print("  -", d)

    return kept


def main() -> None:
    examples = build()
    # Validate every example (integrity + spans + tag well-formedness) before writing.
    for ex in examples:
        ex.validate()
    n = write_jsonl(OUT, examples)
    cats: dict[str, int] = {}
    for ex in examples:
        cats[ex.category] = cats.get(ex.category, 0) + 1
    print(f"wrote {n} held-out-names examples -> {OUT.relative_to(ROOT)}")
    print("by category:", dict(sorted(cats.items())))
    # Report the distinct fresh names used, for the PR body / dataset note.
    names = sorted({ex.ambiguous_token for ex in examples if ex.ambiguous_token})
    print(f"fresh ambiguous names ({len(names)}):", names)


if __name__ == "__main__":
    main()
