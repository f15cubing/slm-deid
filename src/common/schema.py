"""Dataset schema + JSONL (de)serialization + validation (Day 2 foundation).

One JSONL object per example; see docs/tasks/README.md "Example schema". The core invariant,
enforced by :func:`Example.validate`, is::

    unwrap(target) == input          # byte-for-byte
    target is well-formed            # balanced, non-nested tags
    every is_name gold span is tagged in target, and no non-name span is

Gold spans are offsets into the RAW ``input`` string. ``target`` is ``input`` with ``⟨NAME⟩`` tags
around exactly the ``is_name`` spans.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import tags

REGISTERS = {"essay", "dialogue"}
CATEGORIES = {
    "person_vs_eponym",
    "person_vs_place",
    "person_vs_common",
    "first_name_only",
    "possessive",
    "third_party",
    "negative_trap",
    "adversarial",
    "easy",
    "real",  # sourced from a real corpus (mixed categories); tagged per that corpus's labels
}
SOURCES = {
    "synthetic_teacher",
    "presidio_faker",
    "entity_swap",
    "real_crapii",
    "real_tscc",
    "handbuilt",
}


class SchemaError(ValueError):
    """Raised when an example violates the schema or the integrity invariant."""


@dataclass(frozen=True)
class Span:
    """A gold span over the raw ``input`` offsets."""

    start: int
    end: int
    text: str
    is_name: bool

    @staticmethod
    def from_dict(d: dict) -> Span:
        return Span(int(d["start"]), int(d["end"]), str(d["text"]), bool(d["is_name"]))


@dataclass
class Example:
    id: str
    input: str
    target: str
    register: str = "essay"
    category: str = "easy"
    spans: list[Span] = field(default_factory=list)
    source: str = "handbuilt"
    paraphrase_group: str | None = None
    quarantine: bool = False
    # The intended ambiguous surface token this example teaches (e.g. "Austin" for a
    # person_vs_place minimal pair). Set by the minimal-pair generator; enables the
    # category-semantics gate and the token-level eval-leakage guard. Optional/back-compat:
    # legacy rows and negatives leave it None.
    ambiguous_token: str | None = None

    # --- (de)serialization -------------------------------------------------------------
    @staticmethod
    def from_dict(d: dict) -> Example:
        return Example(
            id=str(d["id"]),
            input=str(d["input"]),
            target=str(d["target"]),
            register=str(d.get("register", "essay")),
            category=str(d.get("category", "easy")),
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            source=str(d.get("source", "handbuilt")),
            paraphrase_group=(
                None if d.get("paraphrase_group") is None else str(d["paraphrase_group"])
            ),
            quarantine=bool(d.get("quarantine", False)),
            ambiguous_token=(
                None if d.get("ambiguous_token") is None else str(d["ambiguous_token"])
            ),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    # --- validation --------------------------------------------------------------------
    def name_spans(self) -> list[Span]:
        return [s for s in self.spans if s.is_name]

    def validate(self) -> Example:
        """Raise :class:`SchemaError` on any violation; return self on success."""
        if self.register not in REGISTERS:
            raise SchemaError(f"{self.id}: bad register {self.register!r}")
        if self.category not in CATEGORIES:
            raise SchemaError(f"{self.id}: bad category {self.category!r}")
        if self.source not in SOURCES:
            raise SchemaError(f"{self.id}: bad source {self.source!r}")

        # Integrity + well-formedness of the target.
        if not tags.is_well_formed(self.target):
            raise SchemaError(f"{self.id}: target has malformed/nested tags")
        if tags.unwrap(self.target) != self.input:
            raise SchemaError(f"{self.id}: unwrap(target) != input (integrity)")

        # Gold spans must match the substring they claim to cover.
        for s in self.spans:
            if not (0 <= s.start <= s.end <= len(self.input)):
                raise SchemaError(f"{self.id}: span out of range {s}")
            if self.input[s.start : s.end] != s.text:
                raise SchemaError(
                    f"{self.id}: span text {s.text!r} != input[{s.start}:{s.end}]"
                    f" ({self.input[s.start : s.end]!r})"
                )

        # The set of tagged spans in target must equal exactly the is_name gold spans.
        tagged = {(t.start, t.end, t.text) for t in tags.tagged_spans(self.target)}
        gold = {(s.start, s.end, s.text) for s in self.name_spans()}
        if tagged != gold:
            raise SchemaError(
                f"{self.id}: tagged spans {sorted(tagged)} != gold name spans {sorted(gold)}"
            )
        return self


def loads(line: str) -> Example:
    return Example.from_dict(json.loads(line))


def dumps(ex: Example) -> str:
    return json.dumps(ex.to_dict(), ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[Example]:
    path = Path(path)
    out: list[Example] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(loads(line))
            except (json.JSONDecodeError, KeyError) as e:
                raise SchemaError(f"{path}:{i}: {e}") from e
    return out


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(dumps(ex) + "\n")
            n += 1
    return n


def validate_all(examples: Iterable[Example]) -> list[Example]:
    """Validate every example (raises on the first bad one). Returns the list."""
    return [ex.validate() for ex in examples]


def iter_jsonl(path: str | Path) -> Iterator[Example]:
    yield from read_jsonl(path)
