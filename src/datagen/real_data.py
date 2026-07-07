"""Load real corpora into our Example schema (Day 3 'small real slice').

Currently supports **CRAPII** (Kaggle: "PII Data Detection",
`pii-detection-removal-from-educational-data`), a token-level BIO corpus of ~6.8k student essays.

Format per record: ``document`` (int), ``full_text`` (str), ``tokens`` (list[str]),
``trailing_whitespace`` (list[bool]), ``labels`` (list[str], BIO). PII types:
NAME_STUDENT, EMAIL, USERNAME, ID_NUM, PHONE_NUM, URL_PERSONAL, STREET_ADDRESS.

Conversion policy (see the caveat below):
- ``NAME_STUDENT`` runs -> gold name spans (``is_name=True``), wrapped in ``⟨NAME⟩`` in the target.
- ALL pattern-type labels (EMAIL/PHONE/URL/ID/ADDRESS) -> left UNTAGGED. This is intentional and
  spec-consistent: those belong to regex downstream, so real essays with real-looking emails/phones
  that stay untagged are exactly the negatives we want.

**CAVEAT (documented, not hidden):** CRAPII's ``NAME_STUDENT`` covers *student* names only and
explicitly excludes instructors/authors/other people. Our behavior spec is "tag EVERY person's
name," so an essay mentioning an unlabeled instructor name would teach under-tagging. Mitigation:
keep the real slice SMALL (the synthetic bulk teaches "tag every person"), and flag it in the
dataset card. A higher-fidelity option is to teacher-relabel these essays for all person names.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common import tags
from src.common.schema import Example, Span

NAME_LABEL = "NAME_STUDENT"


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


def _name_char_spans(
    labels: list[str], offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge consecutive NAME_STUDENT B/I tokens into (start, end) char spans."""
    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for (tok_start, tok_end), label in zip(offsets, labels):
        is_name = label.endswith(NAME_LABEL) and label != "O"
        begins = label == f"B-{NAME_LABEL}"
        if is_name and (begins or cur_start is None):
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start, cur_end = tok_start, tok_end
        elif is_name:  # I-NAME_STUDENT continuation
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


def record_to_example(record: dict, id_prefix: str = "crapii") -> Example:
    """Convert one CRAPII record into a validated Example (name spans only)."""
    tokens = record["tokens"]
    trailing = record["trailing_whitespace"]
    labels = record.get("labels") or ["O"] * len(tokens)

    text, offsets = _reconstruct(tokens, trailing)
    char_spans = _name_char_spans(labels, offsets)
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


def load_crapii(
    path: str | Path,
    limit: int | None = None,
    max_chars: int | None = 4000,
    names_only: bool = False,
) -> list[Example]:
    """Load CRAPII ``train.json`` into Examples.

    - ``limit``: cap the number of essays (use a SMALL slice for training; see caveat).
    - ``max_chars``: drop essays longer than this (keeps sequences within the training window).
    - ``names_only``: keep only essays that contain at least one NAME_STUDENT span.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    out: list[Example] = []
    for record in data:
        ex = record_to_example(record)
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
