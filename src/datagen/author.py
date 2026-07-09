"""In-session authored teacher (v3) — a template-based stand-in for the frontier teacher.

Why this exists: the gpt-4o/Anthropic teacher was unavailable for the v3 build (no active billing /
no key), so the passages here are authored in-session from context templates + the eval-disjoint
vocab bank, then routed through the **exact same** quality gate (`quality_gate.filter_examples`),
disposition check (in `generate.build_dataset`), and eval-leakage guards as any teacher output.

It exposes the same interface as :class:`src.datagen.teacher.TeacherGenerator`
(``generate`` / ``generate_pair`` / ``verify_tagging``) so it drops straight into ``build_dataset``.

Design goals, driven by the v2 error analysis (recall/consistency regressed because
`person_vs_common` was 2:1 skewed toward WITHHOLD with little context variety):
- **Person-use is rich and varied**: many educational-context templates where the name is the
  subject *doing something a person does* (lent notes, explained a proof, presented findings), so
  the model learns the CONTEXTUAL cue, not the surface. Minimal pairs are 50/50 by construction, so
  this alone lifts person-use from v2's 18 to hundreds.
- **True minimal-pair contrast**: the non-person half reuses the identical surface in a genuine
  non-person sense (the common word / place / unit), so the model sees `Ruby`-the-person vs
  `ruby`-the-gem, `June`-the-person vs `June`-the-month, etc.

Caveat (recorded in the model/dataset cards): because passages are self-authored, ``verify_tagging``
returns ``None`` — there is no INDEPENDENT second-pass verifier as with the API teacher. The gate
still enforces integrity / well-formedness / schema / category-semantics / disposition, and every
tag is placed by construction, so label trust rests on those deterministic checks.
"""

from __future__ import annotations

import itertools

from src.common import tags
from src.common.schema import Example, Span

# --- person-use context templates (the name is the subject; {name} is the tagged span) --------
# Educational registers. Variety here is the whole point — it teaches the ROLE, not the surface.
PERSON_ESSAY = (
    "{name} stayed after class to help me rework my thesis statement.",
    "During the group project, {name} organized our research into a clear outline.",
    "I asked {name} to review my lab report before I turned it in.",
    "{name} explained the proof again until the whole study group understood it.",
    "Our teacher paired me with {name} for the peer-editing exercise.",
    "{name} volunteered to present the group's findings to the class.",
    "When I missed the lecture, {name} shared detailed notes with me.",
    "{name} caught a mistake in my calculation during the review session.",
    "The tutor said {name} had improved the most on the practice exam.",
    "{name} spent the weekend helping classmates prepare for the midterm.",
    "In seminar, {name} argued that the second source was more reliable.",
    "{name} rebuilt the experiment after the first trial failed.",
)
PERSON_DIALOGUE = (
    "honestly {name} was the one who explained recursion to me last night",
    "thanks to {name} i finally understood the assignment",
    "can you ask {name} to send over the study guide?",
    "{name} said they'd meet us at the library after fourth period",
    "i think {name} already finished the problem set if you want help",
    "{name} walked me through the homework on the bus this morning",
)
POSSESSIVE_PERSON = (
    "{name}'s essay improved dramatically after the peer review.",
    "I borrowed {name}'s notes to study for the quiz.",
    "{name}'s presentation was the clearest one in the class.",
    "The teacher praised {name}'s revision of the introduction.",
    "We followed {name}'s outline to structure the group report.",
)

# --- non-person templates (identical surface, non-person sense; NOTHING tagged) ---------------
PLACE_ESSAY = (
    "We studied the history of {tok} for our geography unit.",
    "The documentary about {tok} covered its rivers and its climate.",
    "Our class mapped the major cities near {tok} for the project.",
    "The population of {tok} grew quickly over the last decade.",
    "For homework we compared the economy of {tok} to its neighbors.",
    "The article described a heat wave that hit {tok} last summer.",
)
PLACE_DIALOGUE = (
    "did you finish the reading about {tok} for geography?",
    "our trip to {tok} got moved to the spring semester",
)
EPONYM_ESSAY = (
    "We applied the {tok} method to approximate the root of the equation.",
    "The result was reported in {tok}s, as the textbook required.",
    "Our physics unit introduced the {tok} law midway through the term.",
    "The {tok} constant showed up throughout the derivation.",
    "The lab measured everything on the {tok} scale for consistency.",
    "The proof relied on the {tok} theorem from the last chapter.",
)
EPONYM_DIALOGUE = (
    "wait do we need the {tok} equation for tomorrow's quiz?",
    "i keep mixing up the {tok} law with the next one",
)
POSSESSIVE_NONPERSON = (
    "{tok}'s law describes the relationship precisely.",
    "We applied {tok}'s method to solve the equation.",
    "{tok}'s constant appears in the final formula.",
    "The chapter derived {tok}'s theorem step by step.",
)

# --- per-token common-word sense for person_vs_common non-person halves -----------------------
# Each value is a natural clause using the surface as its ORDINARY meaning (lowercased where that
# is the common form). The token itself must appear verbatim (category-semantics gate checks it).
COMMON_SENSE = {
    "joy": "a deep sense of joy",
    "melody": "a slow, haunting melody",
    "daisy": "a single daisy in the garden",
    "iris": "an iris blooming beside the path",
    "pearl": "a pearl set in the old necklace",
    "dawn": "the pale light of dawn",
    "autumn": "the cool air of autumn",
    "miles": "several miles down the road",
    "frank": "a frank and honest answer",
    "rich": "a rich, buttery flavor",
    "drew": "she drew a diagram on the board",
    "sky": "the clear evening sky",
    "sunny": "a bright, sunny afternoon",
    "ivy": "ivy climbing the old brick wall",
    "holly": "a sprig of holly on the door",
    "robin": "a robin perched on the fence",
    "jay": "a blue jay in the oak tree",
    "dale": "a green dale between the hills",
    "wade": "we had to wade across the shallow creek",
    "ruby": "a ruby set in the ring",
    "hazel": "a notebook in a deep hazel color",
    "june": "the last week of June",
    "belle": "the belle of the spring dance",
    "lily": "a white lily floating on the pond",
    "wren": "a tiny wren singing at first light",
    "sage": "a pinch of dried sage in the recipe",
    "piper": "a lone piper played at the ceremony",
    "grant": "a research grant funded the new equipment",
    "reed": "a reed swaying at the water's edge",
    "gene": "a single gene controls the trait",
    "buck": "the ticket cost just a buck",
    "chase": "an exciting chase scene in the film",
    "colt": "a young colt in the paddock",
    "dove": "a white dove circled the field",
    "fern": "a fern unfurling in the shade",
    "clay": "a bowl shaped from wet clay",
    "brook": "a brook trickling through the woods",
    "star": "a bright star over the hills",
    "rain": "a light rain fell all afternoon",
    "hart": "a hart grazing at the forest edge",
    "penny": "not a single penny was left",
    "cliff": "a steep cliff above the shore",
    "van": "the delivery van parked outside",
    # senses for the tokens in main's expanded bank (person_vs_common contrast)
    "heath": "open heath stretching to the horizon",
    "bud": "a flower bud about to open",
    "chip": "a chip of blue paint on the sill",
    "ginger": "a knob of fresh ginger",
    "amber": "a piece of amber on the shelf",
    "jade": "a smooth jade carving",
    "olive": "a single olive on the plate",
    "basil": "a few leaves of basil in the pot",
    "hunter": "a jacket in hunter green",
    "lark": "a lark singing over the field",
    "lane": "a narrow country lane",
    "fawn": "a fawn resting in the tall grass",
    "bell": "a brass bell by the door",
    "hank": "a hank of wool on the table",
    "gale": "a fierce gale off the sea",
    "dot": "a small dot of ink on the page",
    "cash": "the stall took only cash",
    "crystal": "a crystal of quartz",
    "coral": "a branch of coral on the reef",
    "rosemary": "a sprig of rosemary",
    "forest": "a dense forest of pine",
}
COMMON_ESSAY = (
    "The poem lingers on {sense} in its closing lines.",
    "In the story, the narrator notices {sense} on the walk home.",
    "The description of {sense} anchored the whole paragraph.",
    "My essay opens with an image of {sense}.",
)
COMMON_DIALOGUE = (
    "the reading had this whole bit about {sense}",
    "i used {sense} as the example in my draft",
)

# --- single-category pools --------------------------------------------------------------------
# Clear personal names, disjoint from the eval blocklist, for first_name_only / third_party / easy.
PERSON_NAMES = (
    "Marcus",
    "Elena",
    "Diego",
    "Aisha",
    "Lena",
    "Kwame",
    "Sofia",
    "Rahul",
    "Nadia",
    "Oscar",
    "Mei",
    "Ibrahim",
    "Carla",
    "Dmitri",
    "Yuki",
    "Fatima",
    "Leo",
    "Hana",
    "Andre",
    "Camila",
)
FIRST_NAME_ONLY = (
    "I sat next to {name} during the whole review session.",
    "{name} reminded me to double-check the citations.",
    "After class {name} offered to quiz me on the vocabulary.",
    "{name} sketched the diagram that finally made it click.",
)
THIRD_PARTY = (
    "My lab partner {name} redid the titration when ours went wrong.",
    "Our teacher {name} extended the deadline for the whole class.",
    "My friend {name} lent me the textbook for the weekend.",
    "The TA, {name}, held an extra session before the exam.",
)
EASY = (
    "Professor {name} opened the lecture with a quick recap.",
    "{name} presented last and answered every question calmly.",
    "The award went to {name} for the best final project.",
    "{name} led the discussion on the assigned chapter.",
)
# Capitalized NON-names: course titles, tools/brands, sentence-initial gerunds. Nothing tagged.
NEGATIVE_TRAP = (
    "Chemistry lab ran late again on Thursday afternoon.",
    "The History essay is due at the start of Monday's class.",
    "Calculus review starts next week before the final.",
    "We built the chart in Excel and pasted it into the report.",
    "I saved the draft in Google Docs so the group could edit it.",
    "Running every morning helped me focus during exams.",
    "Reading the chapter twice made the lab much easier.",
    "Geometry was the hardest unit for me this semester.",
    "The Economics reading connected supply to real prices.",
    "Studying with flashcards finally made the terms stick.",
    "We watched a Spanish film and wrote a short response.",
    "The Physics problem set covered momentum and energy.",
)


def _clean(text: str) -> str:
    return text.strip()


class AuthoredTeacher:
    """Template-based teacher with the TeacherGenerator interface (no network, no API key)."""

    def __init__(self) -> None:
        # independent rotating counters per stream so repeated tokens get varied contexts
        self._c: dict[str, itertools.count] = {}

    def _next(self, key: str) -> int:
        self._c.setdefault(key, itertools.count())
        return next(self._c[key])

    def _pick(self, pool: tuple[str, ...], key: str) -> str:
        return pool[self._next(key) % len(pool)]

    # --- interface parity with TeacherGenerator -----------------------------------------
    def _example(
        self, tagged: str, *, category: str, register: str, id_: str, token: str | None
    ) -> Example:
        target = _clean(tagged)
        raw = tags.unwrap(target)
        spans = [Span(s.start, s.end, s.text, True) for s in tags.tagged_spans(target)]
        return Example(
            id=id_,
            input=raw,
            target=target,
            register=register,
            category=category,
            spans=spans,
            source="synthetic_teacher",
            quarantine=False,
            ambiguous_token=token,
        )

    def _person_passage(self, name_token: str, register: str, *, possessive: bool = False) -> str:
        tagged_name = tags.wrap(name_token)
        if possessive:
            tpl = self._pick(POSSESSIVE_PERSON, "poss_person")
        else:
            tpl = (
                self._pick(PERSON_DIALOGUE, "person_dlg")
                if register == "dialogue"
                else self._pick(PERSON_ESSAY, "person_ess")
            )
        return tpl.format(name=tagged_name)

    def _nonperson_passage(self, token: str, category: str, register: str) -> str:
        if category == "person_vs_place":
            pool = PLACE_DIALOGUE if register == "dialogue" else PLACE_ESSAY
            return self._pick(pool, f"place_{register}").format(tok=token)
        if category == "person_vs_eponym":
            pool = EPONYM_DIALOGUE if register == "dialogue" else EPONYM_ESSAY
            return self._pick(pool, f"epon_{register}").format(tok=token)
        if category == "possessive":  # eponymous possessive (token is an eponym)
            return self._pick(POSSESSIVE_NONPERSON, "poss_np").format(tok=token)
        # person_vs_common: use the token's ordinary sense
        sense = COMMON_SENSE.get(token.lower())
        if sense is None:  # safety net; should not happen (bank ⊆ COMMON_SENSE)
            sense = f"the word {token.lower()}"
        pool = COMMON_DIALOGUE if register == "dialogue" else COMMON_ESSAY
        return self._pick(pool, f"common_{register}").format(sense=sense)

    def generate_pair(
        self,
        category: str,
        *,
        person_token: str,
        nonperson_token: str | None = None,
        register: str = "essay",
        id_prefix: str = "pair",
    ) -> tuple[Example, Example]:
        nonperson_token = nonperson_token or person_token
        is_poss = category == "possessive"
        person = self._example(
            self._person_passage(person_token, register, possessive=is_poss),
            category=category,
            register=register,
            id_=f"{id_prefix}-{person_token}-person",
            token=person_token,
        )
        nonperson = self._example(
            self._nonperson_passage(nonperson_token, category, register),
            category=category,
            register=register,
            id_=f"{id_prefix}-{nonperson_token}-nonperson",
            token=nonperson_token,
        )
        return person, nonperson

    def generate(
        self, category: str, register: str = "essay", id_: str = "gen", token: str | None = None
    ) -> Example:
        if category == "negative_trap":
            tagged = self._pick(NEGATIVE_TRAP, "neg")
        else:
            name = self._pick(PERSON_NAMES, f"name_{category}")
            pool = {"first_name_only": FIRST_NAME_ONLY, "third_party": THIRD_PARTY}.get(
                category, EASY
            )
            tagged = self._pick(pool, category).format(name=tags.wrap(name))
        return self._example(tagged, category=category, register=register, id_=id_, token=token)

    def verify_tagging(self, input_text: str) -> str | None:
        # No independent second pass for in-session authored data (documented caveat).
        return None
