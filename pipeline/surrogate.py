"""Consistent, realistic surrogate replacement (Faker).

Redaction to ``[NAME]`` destroys readability; surrogate replacement swaps each real identifier
for a realistic fake so the text still *reads* like educational prose while carrying no real PII.
Two properties matter:

- **Consistency** — the same original value maps to the same surrogate everywhere in a document
  (so "Sarah … Sarah's essay" stays coreferent), and distinct values get distinct surrogates.
- **Reproducibility** — seeded, so the same input yields the same de-identified output.

Type-aware: a NAME becomes a fake name (matching single-token vs full-name shape), an EMAIL a
fake email, and so on. Faker is already a project dependency (used by the datagen negatives).
"""

from __future__ import annotations


class SurrogateMap:
    """A seeded, per-document map from (label, original) → realistic surrogate.

    Call :meth:`get` for each detected span; repeated calls with the same ``(label, original)``
    return the cached surrogate, so replacement is coreference-preserving.
    """

    def __init__(self, seed: int = 0):
        from faker import Faker

        self._fake = Faker()
        Faker.seed(seed)
        self._cache: dict[tuple[str, str], str] = {}

    def _fresh(self, label: str, original: str) -> str:
        f = self._fake
        if label == "NAME":
            # Preserve shape: a single-token mention → a single given name; else a full name.
            return f.first_name() if len(original.split()) == 1 else f.name()
        if label == "EMAIL":
            return f.email()
        if label == "PHONE":
            return f.numerify("(###) ###-####")
        if label == "SSN":
            return f.numerify("###-##-####")
        if label == "CREDIT_CARD":
            return f.credit_card_number()
        if label == "IP":
            return f.ipv4()
        if label == "URL":
            return f.url()
        if label == "ID":
            return f.numerify("#" * max(4, len(original)))
        return f"[{label}]"

    def get(self, label: str, original: str) -> str:
        key = (label, original)
        if key not in self._cache:
            self._cache[key] = self._fresh(label, original)
        return self._cache[key]
