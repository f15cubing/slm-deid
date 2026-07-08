#!/usr/bin/env python3
"""Generate *co-occurrence* contrast examples: the SAME ambiguous surface used as a
person (tagged) AND as the thing (untagged) **in one passage** — e.g. "Ivy" the
person next to "ivy" the plant.

Motivation: the v2 minimal pairs put the two senses in *separate* passages. That
teaches the token-in-isolation contrast but never forces the model to disambiguate two
uses of the same surface within a single context — which is where over-tagging (the
Day-3 regression) actually bites. These single-passage co-occurrence examples give that
signal directly, as a seed batch for the next data-gen run.

Authoring format (per passage):
    [[Name]]   → the PERSON occurrence  → tagged ⟨NAME⟩ + gold span is_name=True
    {{word}}   → the THING occurrence   → left untagged + gold span is_name=False
    everything else is literal text.

Offsets and tags are computed deterministically from that markup, so spans are correct
by construction — no teacher LLM, nothing to hand-verify. Every row is then run through
the real schema validator and the same leakage guard the merge pipeline uses
(``vocab.blocklist_surfaces_in``), so this batch is safe to fold into the next merge:

    python -m src.datagen.merge --sources \\
        data/splits/train.jsonl data/splits/val.jsonl data/cooccur/cooccur.jsonl \\
        --out data

Run: ``python scripts/gen_cooccurrence.py`` → writes data/cooccur/cooccur.jsonl
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.common import schema, tags  # noqa: E402
from src.common.schema import Example, Span  # noqa: E402
from src.datagen import vocab  # noqa: E402

OUT = REPO / "data" / "cooccur" / "cooccur.jsonl"

_MARK = re.compile(r"\[\[(.+?)\]\]|\{\{(.+?)\}\}")


def blocklist_surfaces_in(text: str) -> set[str]:
    """Eval surfaces present in ``text`` (case-insensitive, whole-word).

    Prefers the pipeline's own guard when the installed ``vocab`` has it (user's branch);
    falls back to a local scan over ``vocab.BLOCKLIST`` so this script also runs on older
    branches where that helper doesn't exist yet.
    """
    fn = getattr(vocab, "blocklist_surfaces_in", None)
    if callable(fn):
        return set(fn(text))
    low = text.lower()
    hits: set[str] = set()
    for surface in vocab.BLOCKLIST:
        if re.search(r"\b" + re.escape(surface.lower()) + r"\b", low):
            hits.add(surface)
    return hits


def build(marked: str) -> tuple[str, str, list[Span]]:
    """Turn an authoring string into (input, target, spans).

    ``[[x]]`` → person span (is_name=True, wrapped in target).
    ``{{x}}`` → thing span  (is_name=False, left as plain text in target).
    """
    clean = ""
    spans: list[Span] = []
    i = 0
    for m in _MARK.finditer(marked):
        clean += marked[i : m.start()]
        is_name = m.group(1) is not None
        inner = m.group(1) if is_name else m.group(2)
        start = len(clean)
        clean += inner
        spans.append(Span(start=start, end=len(clean), text=inner, is_name=is_name))
        i = m.end()
    clean += marked[i:]

    # Rebuild target: wrap only the is_name spans; everything else byte-identical to input.
    out: list[str] = []
    cur = 0
    for s in sorted(spans, key=lambda s: s.start):
        out.append(clean[cur : s.start])
        out.append(tags.wrap(s.text) if s.is_name else clean[s.start : s.end])
        cur = s.end
    out.append(clean[cur:])
    return clean, "".join(out), spans


# (id, category, register, ambiguous_token, marked passage)
# Person occurrence = [[...]]; thing occurrence = {{...}}. Tokens are drawn only from the
# eval-DISJOINT bank in src/datagen/vocab.py.
PASSAGES: list[tuple[str, str, str, str, str]] = [
    # ---- person_vs_common: given name vs the plant/flower/word it collides with ----
    ("cooccur-ivy-1", "person_vs_common", "essay", "Ivy",
     "When [[Ivy]] moved into the old cottage, she was delighted to find {{ivy}} "
     "climbing the stone walls, its dark leaves softening the weathered facade."),
    ("cooccur-holly-1", "person_vs_common", "essay", "Holly",
     "[[Holly]] volunteered to decorate the hall, weaving sprigs of {{holly}} with "
     "their bright red berries into every wreath along the staircase."),
    ("cooccur-iris-1", "person_vs_common", "essay", "Iris",
     "Every spring [[Iris]] planted a fresh row of {{iris}} along the back fence, and "
     "by June the purple blooms had all but taken over the garden."),
    ("cooccur-daisy-1", "person_vs_common", "dialogue", "Daisy",
     "\"Look what I found,\" said [[Daisy]], holding up a single white {{daisy}} she had "
     "picked from the edge of the lawn."),
    ("cooccur-pearl-1", "person_vs_common", "essay", "Pearl",
     "[[Pearl]] inherited her grandmother's necklace, a single {{pearl}} on a fine gold "
     "chain that she wore to every recital."),
    ("cooccur-dawn-1", "person_vs_common", "essay", "Dawn",
     "[[Dawn]] loved to hike before {{dawn}}, reaching the summit just as the first light "
     "spilled over the ridge."),
    ("cooccur-autumn-1", "person_vs_common", "essay", "Autumn",
     "By the time {{autumn}} arrived, [[Autumn]] had already filled three sketchbooks with "
     "drawings of the turning leaves."),
    ("cooccur-robin-1", "person_vs_common", "dialogue", "Robin",
     "\"Did you name the cat after the bird?\" I asked. [[Robin]] laughed and said no — "
     "the {{robin}} on the feeder had come first, long before the cat."),
    ("cooccur-jay-1", "person_vs_common", "essay", "Jay",
     "[[Jay]] kept a careful journal of every {{jay}} that visited the oak, noting the flash "
     "of blue each time one landed on the rail."),
    ("cooccur-sky-1", "person_vs_common", "essay", "Sky",
     "[[Sky]] repainted the {{sky}} a dozen different shades before finally settling on the "
     "pale gold of a late summer evening."),
    ("cooccur-melody-1", "person_vs_common", "essay", "Melody",
     "[[Melody]] hummed the {{melody}} over and over until the whole choir had it by heart."),
    ("cooccur-joy-1", "person_vs_common", "essay", "Joy",
     "Nothing could dim the {{joy}} on the crowded platform when [[Joy]] finally stepped off "
     "the train after a year abroad."),
    ("cooccur-frank-1", "person_vs_common", "dialogue", "Frank",
     "\"Let me be {{frank}} with you,\" [[Frank]] said, setting down his coffee. \"The numbers "
     "just don't add up.\""),
    ("cooccur-rich-1", "person_vs_common", "essay", "Rich",
     "[[Rich]] never acted {{rich}}, even after the company sold and the money finally came in."),
    ("cooccur-miles-1", "person_vs_common", "essay", "Miles",
     "[[Miles]] had walked for {{miles}} before he realized he had left the map back at the "
     "trailhead."),
    ("cooccur-drew-1", "person_vs_common", "essay", "Drew",
     "[[Drew]] {{drew}} the same lighthouse every summer, filling one notebook after another."),
    ("cooccur-dale-1", "person_vs_common", "essay", "Dale",
     "[[Dale]] grew up in a quiet {{dale}} in the north, where the hills folded down into a "
     "single green valley."),
    ("cooccur-wade-1", "person_vs_common", "dialogue", "Wade",
     "[[Wade]] watched the children {{wade}} into the shallows, their laughter carrying clear "
     "across the lake."),
    ("cooccur-sunny-1", "person_vs_common", "essay", "Sunny",
     "[[Sunny]] promised the picnic would go ahead on any {{sunny}} afternoon, rain the day "
     "before be damned."),
    # ---- person_vs_place: given name vs the city/place it collides with ----
    ("cooccur-austin-1", "person_vs_place", "essay", "Austin",
     "[[Austin]] had never actually been to {{Austin}}, a coincidence his coworkers in Texas "
     "never let him forget."),
    ("cooccur-sydney-1", "person_vs_place", "essay", "Sydney",
     "[[Sydney]] booked the flight to {{Sydney}} months in advance, eager to finally see the "
     "harbor she had only known from postcards."),
    ("cooccur-phoenix-1", "person_vs_place", "essay", "Phoenix",
     "After the wildfire, [[Phoenix]] moved back to {{Phoenix}} to help her family rebuild the "
     "house from the ground up."),
    ("cooccur-savannah-1", "person_vs_place", "essay", "Savannah",
     "[[Savannah]] wrote her entire thesis on the {{savannah}}, though she had never once set "
     "foot outside the city."),
    ("cooccur-brooklyn-1", "person_vs_place", "dialogue", "Brooklyn",
     "\"People are always surprised,\" [[Brooklyn]] said, \"to learn I grew up nowhere near "
     "{{Brooklyn}} and have never crossed the bridge I was named for.\""),
    ("cooccur-salem-1", "person_vs_place", "essay", "Salem",
     "[[Salem]] spent the whole drive telling ghost stories about {{Salem}}, where the family "
     "was headed for the long weekend."),
    # ---- person_vs_eponym: surname vs the unit named after someone ----
    ("cooccur-watt-1", "person_vs_eponym", "essay", "Watt",
     "[[Watt]] explained to the class that a single sixty-{{watt}} bulb would be plenty of "
     "light for such a small study."),
    ("cooccur-joule-1", "person_vs_eponym", "essay", "Joule",
     "[[Joule]] reminded the students that one {{joule}} is a surprisingly small amount of "
     "energy in everyday terms."),
    ("cooccur-hertz-1", "person_vs_eponym", "essay", "Hertz",
     "[[Hertz]] tuned the string patiently until the meter read exactly four hundred and forty "
     "{{hertz}}."),
    ("cooccur-kelvin-1", "person_vs_eponym", "dialogue", "Kelvin",
     "\"Set it to two hundred and fifty {{kelvin}},\" [[Kelvin]] said, \"and label every sample "
     "before you close the freezer.\""),
]


def main() -> None:
    examples: list[Example] = []
    problems: list[str] = []
    for id_, category, register, token, marked in PASSAGES:
        clean, target, spans = build(marked)
        ex = Example(
            id=id_,
            input=clean,
            target=target,
            register=register,
            category=category,
            spans=spans,
            source="handbuilt",
            paraphrase_group=None,
            quarantine=False,
            ambiguous_token=token,
        )
        # --- self-checks (fail loud; we want this batch to be clean by construction) ---
        try:
            ex.validate()
        except schema.SchemaError as e:
            problems.append(f"{id_}: SCHEMA {e}")
            continue
        has_person = any(s.is_name for s in spans)
        has_thing = any(not s.is_name for s in spans)
        if not (has_person and has_thing):
            problems.append(f"{id_}: not a co-occurrence (person={has_person} thing={has_thing})")
        if token.lower() not in clean.lower():
            problems.append(f"{id_}: ambiguous_token {token!r} absent from passage")
        leaked = blocklist_surfaces_in(clean)
        if leaked:
            problems.append(f"{id_}: LEAKAGE — eval surface(s) {sorted(leaked)} in passage")
        examples.append(ex)

    if problems:
        print("REFUSING TO WRITE — problems found:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        sys.exit(1)

    n = schema.write_jsonl(OUT, examples)
    by_cat: dict[str, int] = {}
    by_reg: dict[str, int] = {}
    for ex in examples:
        by_cat[ex.category] = by_cat.get(ex.category, 0) + 1
        by_reg[ex.register] = by_reg.get(ex.register, 0) + 1
    print(f"wrote {n} co-occurrence examples -> {OUT.relative_to(REPO)}")
    print(f"  categories: {by_cat}")
    print(f"  registers : {by_reg}")
    print("  all rows pass schema.validate + leakage guard.")


if __name__ == "__main__":
    main()
