"""Pattern-type NEGATIVES via Faker (Day 2, spec S2.9).

The model's job is NAME judgment; emails / phones / IDs / URLs are owned by regex downstream, so
the training data must teach the model to **leave them untagged**. These generators emit passages
that contain such pattern-type PII with **zero name tags** (pure negatives), plus mixed items where
a real name IS tagged next to an untagged email/phone (teaching the boundary).

Uses Faker directly (light, deterministic with a seed). Presidio's Sentence Faker is a drop-in
upgrade for exact-span pattern labels if we later want them; for NEGATIVES we only need the tokens
present and untagged.
"""

from __future__ import annotations

from src.common import tags
from src.common.schema import Example, Span

# Templates with no person name -> nothing should be tagged. {pii} is an email/phone/id/url.
_PURE_TEMPLATES = [
    "Please submit the form and email me at {pii} if you have questions.",
    "You can reach the help desk at {pii} during office hours.",
    "My student ID is {pii} for the records.",
    "The syllabus is posted at {pii} for everyone to download.",
    "Call {pii} to reschedule the tutoring session.",
]

# Templates with exactly one person name (tagged) next to untagged pattern PII.
_MIXED_TEMPLATES = [
    "Thanks {name} — you can email me the notes at {pii}.",
    "{name} said the assignment link is {pii}.",
    "Ask {name} to text me at {pii} before class.",
]


def _pii_values(fake) -> list[str]:
    return [fake.email(), fake.phone_number(), fake.bothify("ID-#####"), fake.url()]


def generate_negatives(n: int = 20, seed: int = 0) -> list[Example]:
    """Generate ``n`` pattern-type negative/mixed examples (deterministic given ``seed``)."""
    from faker import Faker

    fake = Faker()
    Faker.seed(seed)

    examples: list[Example] = []
    i = 0
    while len(examples) < n:
        i += 1
        pii = _pii_values(fake)[i % 4]
        if i % 3 == 0:
            # mixed: one tagged name + untagged pii
            name = fake.first_name()
            tmpl = _MIXED_TEMPLATES[i % len(_MIXED_TEMPLATES)]
            raw = tmpl.format(name=name, pii=pii)
            start = raw.index(name)
            target = raw[:start] + tags.wrap(name) + raw[start + len(name):]
            spans = [Span(start, start + len(name), name, True)]
            category = "negative_trap"
        else:
            # pure negative: pattern PII only, nothing tagged
            tmpl = _PURE_TEMPLATES[i % len(_PURE_TEMPLATES)]
            raw = tmpl.format(pii=pii)
            target = raw
            spans = []
            category = "negative_trap"

        ex = Example(
            id=f"neg-{seed}-{i:04d}",
            input=raw,
            target=target,
            register="dialogue" if i % 2 else "essay",
            category=category,
            spans=spans,
            source="presidio_faker",
            quarantine=False,
        )
        # Guard: a name string must not accidentally collide with the pii token.
        try:
            examples.append(ex.validate())
        except Exception:
            continue
    return examples
