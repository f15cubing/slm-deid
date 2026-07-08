"""Author the quarantined hard-cases eval set (Day 2, spec S2.6).

Each scenario is written as inline-tagged text (persons wrapped in ⟨NAME⟩…⟨/NAME⟩); the builder
derives `input` (unwrap), `target` (the tagged text), and the gold name `spans` (offsets computed,
never hand-typed), then validates every example. Output: eval/hardcases/hardcases.jsonl.

This set is QUARANTINED: source="handbuilt", quarantine=true, and it must NEVER be fed to the
data-gen teacher, augmentation, or the training splits (enforced by tests/test_no_eval_leakage.py).

Categories mirror docs/tasks/README.md. Grow toward 80-150 by mining real CRAPII/TSCC later.
Run: PYTHONPATH=. python scripts/build_hardcases.py
"""

from __future__ import annotations

from pathlib import Path

from src.common import tags
from src.common.schema import Example, Span, write_jsonl

OUT = Path("eval/hardcases/hardcases.jsonl")

# (id, category, register, inline-tagged text, paraphrase_group)
# Persons are wrapped; every other capitalized token is deliberately left untagged.
SCENARIOS: list[tuple[str, str, str, str, str | None]] = [
    # --- person vs eponym/method -------------------------------------------------------
    (
        "eponym-newton-person-1",
        "person_vs_eponym",
        "essay",
        f"{tags.wrap('Newton')} stayed late in the lab because the experiment kept failing.",
        "newton-person",
    ),
    (
        "eponym-newton-person-2",
        "person_vs_eponym",
        "dialogue",
        f"honestly {tags.wrap('Newton')} was the one who fixed my code last night",
        "newton-person",
    ),
    (
        "eponym-newton-method-1",
        "person_vs_eponym",
        "essay",
        "We applied the Newton method to approximate the root of the equation.",
        "newton-method",
    ),
    (
        "eponym-newton-method-2",
        "person_vs_eponym",
        "essay",
        "Using Newton's method, the solver converged after four iterations.",
        "newton-method",
    ),
    (
        "eponym-gauss-unit",
        "person_vs_eponym",
        "essay",
        "The magnetic field strength was measured in gauss during the experiment.",
        None,
    ),
    (
        "eponym-pascal-unit",
        "person_vs_eponym",
        "essay",
        "Atmospheric pressure is often reported in pascals rather than bars.",
        None,
    ),
    (
        "eponym-turing-test",
        "person_vs_eponym",
        "essay",
        "The chatbot failed the Turing test in under a minute.",
        "turing",
    ),
    (
        "eponym-turing-person",
        "person_vs_eponym",
        "essay",
        f"{tags.wrap('Turing')} presented his proof to the seminar on Friday.",
        "turing",
    ),
    # --- person vs place ---------------------------------------------------------------
    (
        "place-chelsea-person-1",
        "person_vs_place",
        "dialogue",
        f"{tags.wrap('Chelsea')} helped me revise my thesis statement after class.",
        "chelsea",
    ),
    (
        "place-chelsea-person-2",
        "person_vs_place",
        "dialogue",
        f"thanks to {tags.wrap('Chelsea')} I finally understood the proof",
        "chelsea",
    ),
    (
        "place-chelsea-place",
        "person_vs_place",
        "essay",
        "Last summer I visited Chelsea and walked along the river all afternoon.",
        "chelsea",
    ),
    (
        "place-darwin-city",
        "person_vs_place",
        "essay",
        "Our flight to Darwin was delayed by a tropical storm.",
        "darwin",
    ),
    (
        "place-darwin-person",
        "person_vs_place",
        "essay",
        f"{tags.wrap('Darwin')} sat next to me and shared his lab notes.",
        "darwin",
    ),
    (
        "place-florence-city",
        "person_vs_place",
        "essay",
        "The Renaissance art in Florence left a deep impression on the class.",
        "florence",
    ),
    (
        "place-florence-person",
        "person_vs_place",
        "dialogue",
        f"{tags.wrap('Florence')} said she would email the study guide tonight",
        "florence",
    ),
    (
        "place-jordan-country",
        "person_vs_place",
        "essay",
        "We studied the water scarcity crisis in Jordan for our geography unit.",
        "jordan",
    ),
    (
        "place-jordan-person",
        "person_vs_place",
        "dialogue",
        f"{tags.wrap('Jordan')} lent me a charger during the exam review",
        "jordan",
    ),
    (
        "place-madison-city",
        "person_vs_place",
        "essay",
        "The conference was held in Madison over spring break.",
        None,
    ),
    # --- person vs common word ---------------------------------------------------------
    (
        "common-grace-person",
        "person_vs_common",
        "dialogue",
        f"{tags.wrap('Grace')} lent me her notes before the midterm.",
        "grace",
    ),
    (
        "common-grace-concept",
        "person_vs_common",
        "essay",
        "She handled the criticism with grace and kept improving her draft.",
        "grace",
    ),
    (
        "common-hope-person",
        "person_vs_common",
        "dialogue",
        f"{tags.wrap('Hope')} explained the assignment again after class",
        "hope",
    ),
    (
        "common-hope-concept",
        "person_vs_common",
        "essay",
        "There is still hope that the project will be finished on time.",
        "hope",
    ),
    (
        "common-faith-concept",
        "person_vs_common",
        "essay",
        "The argument rests on faith rather than evidence.",
        "faith",
    ),
    (
        "common-faith-person",
        "person_vs_common",
        "dialogue",
        f"{tags.wrap('Faith')} asked a sharp question during the seminar",
        "faith",
    ),
    (
        "common-bishop-chess",
        "person_vs_common",
        "essay",
        "I moved my bishop to threaten the queen in the endgame.",
        "bishop",
    ),
    (
        "common-bishop-person",
        "person_vs_common",
        "dialogue",
        f"our TA {tags.wrap('Bishop')} regraded my problem set",
        "bishop",
    ),
    (
        "common-rose-flower",
        "person_vs_common",
        "essay",
        "The rose in the courtyard bloomed early this year.",
        "rose",
    ),
    (
        "common-rose-person",
        "person_vs_common",
        "dialogue",
        f"{tags.wrap('Rose')} sat with me while I debugged the loop",
        "rose",
    ),
    (
        "common-may-month",
        "person_vs_common",
        "essay",
        "The final report is due in May before the term ends.",
        "may",
    ),
    (
        "common-may-person",
        "person_vs_common",
        "dialogue",
        f"{tags.wrap('May')} volunteered to present the group's findings",
        "may",
    ),
    (
        "common-mark-noun",
        "person_vs_common",
        "essay",
        "She left a mark on the whiteboard that no one could erase.",
        "mark",
    ),
    (
        "common-mark-person",
        "person_vs_common",
        "dialogue",
        f"{tags.wrap('Mark')} reviewed my essay and suggested a stronger thesis",
        "mark",
    ),
    (
        "common-baker-job",
        "person_vs_common",
        "essay",
        "The baker down the street sponsored our fundraising bake sale.",
        "baker",
    ),
    (
        "common-baker-person",
        "person_vs_common",
        "essay",
        f"Professor {tags.wrap('Baker')} extended the deadline for the whole class.",
        "baker",
    ),
    # --- first-name-only / buried surname ----------------------------------------------
    (
        "firstname-sam",
        "first_name_only",
        "dialogue",
        f"thanks, {tags.wrap('Sam')} — that explanation finally clicked",
        None,
    ),
    (
        "firstname-buried-alvarez",
        "first_name_only",
        "essay",
        f"When {tags.wrap('Alvarez')} presented, the room finally went quiet.",
        None,
    ),
    (
        "firstname-liang",
        "first_name_only",
        "dialogue",
        f"ok {tags.wrap('Liang')} i'll send you the notes in a sec",
        None,
    ),
    # --- possessive / inflected --------------------------------------------------------
    (
        "poss-sarah-person",
        "possessive",
        "essay",
        f"{tags.wrap('Sarah')}'s essay improved dramatically after peer review.",
        "poss",
    ),
    (
        "poss-newton-laws",
        "possessive",
        "essay",
        "Newton's laws describe the motion of classical objects.",
        "poss",
    ),
    (
        "poss-ohms-law",
        "possessive",
        "essay",
        "We verified Ohm's law using three different resistors.",
        None,
    ),
    # --- third parties -----------------------------------------------------------------
    (
        "third-teacher-rivera",
        "third_party",
        "essay",
        f"My teacher, Ms. {tags.wrap('Rivera')}, said the deadline moved to Monday.",
        None,
    ),
    (
        "third-friend-omar",
        "third_party",
        "dialogue",
        f"my friend {tags.wrap('Omar')} already finished the lab report",
        None,
    ),
    (
        "third-parent",
        "third_party",
        "essay",
        f"My mom, {tags.wrap('Priya')}, quizzed me on vocabulary every evening.",
        None,
    ),
    # --- negative traps (capitalized non-names, brands, course titles) -----------------
    (
        "neg-course-title",
        "negative_trap",
        "essay",
        "Introduction to Psychology met every Tuesday in the east wing.",
        None,
    ),
    (
        "neg-sentence-initial",
        "negative_trap",
        "essay",
        "Biology quickly became my favorite subject this semester.",
        None,
    ),
    (
        "neg-brand",
        "negative_trap",
        "dialogue",
        "i wrote the whole draft on my Android phone during the bus ride",
        None,
    ),
    (
        "neg-month-day",
        "negative_trap",
        "essay",
        "March felt endless, but April brought the final presentations.",
        None,
    ),
    (
        "neg-org",
        "negative_trap",
        "essay",
        "The Red Cross visited our school to run a first-aid workshop.",
        None,
    ),
    # --- easy controls (a prompted base should get these) ------------------------------
    (
        "easy-dear-johnson",
        "easy",
        "essay",
        f"Dear Mr. {tags.wrap('Johnson')}, thank you for reviewing my application.",
        None,
    ),
    (
        "easy-two-names",
        "easy",
        "dialogue",
        f"{tags.wrap('Ada')} and {tags.wrap('Alan')} paired up for the coding project.",
        None,
    ),
    ("easy-signed", "easy", "essay", f"Sincerely, {tags.wrap('Maria Gonzalez')}", None),
]


def build() -> list[Example]:
    examples: list[Example] = []
    for id_, category, register, tagged, pg in SCENARIOS:
        raw = tags.unwrap(tagged)
        spans = [Span(s.start, s.end, s.text, True) for s in tags.tagged_spans(tagged)]
        ex = Example(
            id=id_,
            input=raw,
            target=tagged,
            register=register,
            category=category,
            spans=spans,
            source="handbuilt",
            paraphrase_group=pg,
            quarantine=True,
        )
        examples.append(ex.validate())
    return examples


def main() -> None:
    examples = build()
    n = write_jsonl(OUT, examples)
    cats: dict[str, int] = {}
    for ex in examples:
        cats[ex.category] = cats.get(ex.category, 0) + 1
    print(f"wrote {n} examples -> {OUT}")
    for cat in sorted(cats):
        print(f"  {cat}: {cats[cat]}")


if __name__ == "__main__":
    main()
