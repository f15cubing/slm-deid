"""Load real corpora into our Example schema (Day 3 'small real slice').

Supports **CRAPII** in two shapes:
- the Kaggle *dataset* `langdonholmes/cleaned-repository-of-annotated-pii` — JSON **Lines**, name
  label ``NAME`` (covers person names generally), 14 PII types;
- the Kaggle *competition* `pii-detection-removal-from-educational-data` — a JSON **array**, name
  label ``NAME_STUDENT`` (student names only), 7 PII types.

Format per record (both): ``document``, ``full_text``, ``tokens`` (list[str]),
``trailing_whitespace`` (list[bool]), ``labels`` (list[str], BIO).

Conversion policy:
- Name runs (``B-/I-`` with a type in :data:`NAME_TYPES`) -> gold name spans (``is_name=True``),
  wrapped in ``⟨NAME⟩`` in the target.
- ALL pattern-type labels (EMAIL/PHONE/URL/ID/ADDRESS/etc.) -> left UNTAGGED. Intentional and
  spec-consistent: those belong to regex downstream, so real essays with real-looking emails/phones
  that stay untagged are exactly the negatives we want.

**CAVEAT (documented, not hidden):** the competition's ``NAME_STUDENT`` excludes instructors/
authors/other people; the dataset's ``NAME`` is broader but still corpus-defined. Our spec is "tag
EVERY person's name," so keep the real slice SMALL (the synthetic bulk teaches "tag every person")
and note it in the dataset card. Higher-fidelity option: teacher-relabel these essays.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common import tags
from src.common.schema import Example, Span

# Entity types (after the B-/I- prefix) that count as a person name across CRAPII variants.
NAME_TYPES = {"NAME", "NAME_STUDENT"}


def _reconstruct(tokens: list[str], trailing: list[bool]) -> tuple[str, list[tuple[int, int]]]:
    """Rebuild text and each token's (start, end) char offsets from tokens + whitespace flags."""
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for tok, ws in zip(tokens, trailing):
        start = pos
        text_parts.append(tok)
        pos += len(tok)
        offsets.append((start, pos))
        if ws:
            text_parts.append(" ")
            pos += 1
    return "".join(text_parts), offsets


def _parse_label(label: str) -> tuple[str, str]:
    """Return (prefix, entity_type) for a BIO label. 'O' -> ('O', '')."""
    if not label or label == "O":
        return "O", ""
    prefix, _, etype = label.partition("-")
    return prefix, etype.strip()


def _name_char_spans(
    labels: list[str], offsets: list[tuple[int, int]], name_types: set[str]
) -> list[tuple[int, int]]:
    """Merge consecutive name B/I tokens into (start, end) char spans."""
    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for (tok_start, tok_end), label in zip(offsets, labels):
        prefix, etype = _parse_label(label)
        is_name = etype in name_types
        begins = prefix == "B"
        if is_name and (begins or cur_start is None):
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start, cur_end = tok_start, tok_end
        elif is_name:  # I- continuation
            cur_end = tok_end
        else:
            if cur_start is not None:
                spans.append((cur_start, cur_end))
                cur_start = cur_end = None
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans


def _build_target(text: str, spans: list[tuple[int, int]]) -> str:
    out: list[str] = []
    cursor = 0
    for s, e in spans:
        out.append(text[cursor:s])
        out.append(tags.wrap(text[s:e]))
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


def record_to_example(
    record: dict, id_prefix: str = "crapii", name_types: set[str] = NAME_TYPES
) -> Example:
    """Convert one CRAPII record into a validated Example (name spans only)."""
    tokens = record["tokens"]
    trailing = record.get("trailing_whitespace") or [True] * len(tokens)
    labels = record.get("labels") or ["O"] * len(tokens)

    text, offsets = _reconstruct(tokens, trailing)
    char_spans = _name_char_spans(labels, offsets, name_types)
    target = _build_target(text, char_spans)
    spans = [Span(s, e, text[s:e], True) for s, e in char_spans]

    return Example(
        id=f"{id_prefix}-{record.get('document', 'x')}",
        input=text,
        target=target,
        register="essay",
        category="real",
        spans=spans,
        source="real_crapii",
        quarantine=False,
    )


def _read_records(path: str | Path) -> list[dict]:
    """Read a CRAPII file as either a JSON array or JSON Lines."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_crapii(
    path: str | Path,
    limit: int | None = None,
    max_chars: int | None = 4000,
    names_only: bool = False,
    name_types: set[str] = NAME_TYPES,
) -> list[Example]:
    """Load CRAPII (JSON array or JSON Lines) into Examples.

    - ``limit``: cap the number of essays (use a SMALL slice for training; see caveat).
    - ``max_chars``: drop essays longer than this (keeps sequences within the training window).
    - ``names_only``: keep only essays that contain at least one name span.
    - ``name_types``: entity types treated as person names (``NAME`` and/or ``NAME_STUDENT``).
    """
    out: list[Example] = []
    for record in _read_records(path):
        ex = record_to_example(record, name_types=name_types)
        if max_chars is not None and len(ex.input) > max_chars:
            continue
        if names_only and not ex.name_spans():
            continue
        try:
            out.append(ex.validate())
        except Exception:
            continue  # skip any record whose offsets don't reconstruct cleanly
        if limit is not None and len(out) >= limit:
            break
    return out
