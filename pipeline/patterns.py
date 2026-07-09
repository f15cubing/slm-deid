"""Deterministic pattern-type PII detection (regex; Presidio optional).

Per the BrainLift/plan split of labor: **pattern-type identifiers** — emails, phone numbers,
SSNs, credit cards, IPs, URLs, long ID numbers — are handled *deterministically* here, not by
the model. The model's scarce capacity is spent only on the judgment a regex can't make
(is this token a person's name?).

The default backend is a compact, dependency-free regex set so the pipeline (and its tests) run
hermetically on CPU with no model download. Presidio's predefined pattern recognizers are
available as an optional higher-recall backend (``backend="presidio"``); they are invoked
*without* an NLP engine, so no spaCy model is required — and if Presidio is unavailable we fall
back to regex rather than fail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: earlier, more specific patterns win overlap resolution against later, broader
# ones (e.g. an email containing a dotted host beats the bare IP pattern; SSN beats generic ID).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b\d(?:[ \-]?\d){12,15}\b")),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)(?:\+?1[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}(?!\d)",
        ),
    ),
    # Dotted-quad with each octet bounded 0-255 (rejects e.g. "999.1.1.1"). Version strings like
    # "1.2.3.4" are genuinely IP-shaped and can't be disambiguated without more context.
    (
        "IP",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
    ),
    # Generic ID: an alphanumeric token that mixes letters AND digits (student/record IDs like
    # "STU00081234"). Requiring a letter avoids tagging ordinary numbers (scores, populations,
    # decimals) — those are not identifiers. Last, so the specific patterns above win overlaps.
    ("ID", re.compile(r"\b(?=[A-Za-z0-9\-]*[A-Za-z])(?=[A-Za-z0-9\-]*\d)[A-Za-z0-9\-]{5,}\b")),
]


@dataclass(frozen=True)
class PatternSpan:
    """A deterministically-detected pattern-PII span over the original text offsets."""

    start: int
    end: int
    text: str
    label: str  # EMAIL | URL | SSN | CREDIT_CARD | PHONE | IP | ID


def _resolve_overlaps(spans: list[PatternSpan]) -> list[PatternSpan]:
    """Greedy left-to-right, longest-first: accept a span only if it doesn't overlap a kept one.

    Ties (same start) resolve to the longer match; genuine overlaps to the earlier start. This
    keeps, e.g., a full email rather than the bare domain a later pattern might re-match.
    """
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[PatternSpan] = []
    for sp in ordered:
        if any(sp.start < k.end and k.start < sp.end for k in kept):
            continue
        kept.append(sp)
    return sorted(kept, key=lambda s: s.start)


def _detect_regex(text: str) -> list[PatternSpan]:
    found: list[PatternSpan] = []
    for label, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            # Trim trailing punctuation the greedy URL/ID patterns may swallow.
            while e > s and text[e - 1] in ".,;:!?)]}\"'":
                e -= 1
            if e > s:
                found.append(PatternSpan(s, e, text[s:e], label))
    return _resolve_overlaps(found)


def _detect_presidio(text: str) -> list[PatternSpan]:
    """Higher-recall backend using Presidio's predefined *pattern* recognizers (no NLP engine).

    Each recognizer is a plain regex/checksum matcher; calling ``.analyze(..., nlp_artifacts=
    None)`` avoids loading spaCy. Any import/version issue falls back to the regex backend.
    """
    try:
        from presidio_analyzer.predefined_recognizers import (
            CreditCardRecognizer,
            EmailRecognizer,
            IpRecognizer,
            PhoneRecognizer,
            UrlRecognizer,
            UsSsnRecognizer,
        )
    except Exception:  # pragma: no cover - optional dependency / version drift
        return _detect_regex(text)

    recognizers = [
        ("EMAIL", EmailRecognizer(), ["EMAIL_ADDRESS"]),
        ("URL", UrlRecognizer(), ["URL"]),
        ("SSN", UsSsnRecognizer(), ["US_SSN"]),
        ("CREDIT_CARD", CreditCardRecognizer(), ["CREDIT_CARD"]),
        ("PHONE", PhoneRecognizer(), ["PHONE_NUMBER"]),
        ("IP", IpRecognizer(), ["IP_ADDRESS"]),
    ]
    found: list[PatternSpan] = []
    for label, rec, entities in recognizers:
        try:
            results = rec.analyze(text=text, entities=entities, nlp_artifacts=None) or []
        except Exception:  # pragma: no cover - be resilient to recognizer signature drift
            continue
        for r in results:
            found.append(PatternSpan(r.start, r.end, text[r.start : r.end], label))
    # Fold in the regex-only ID pattern (Presidio has no generic student-ID recognizer).
    found += [s for s in _detect_regex(text) if s.label == "ID"]
    return _resolve_overlaps(found)


def detect(text: str, backend: str = "regex") -> list[PatternSpan]:
    """Detect pattern-type PII spans in ``text``.

    ``backend="regex"`` (default) is hermetic and dependency-free. ``backend="presidio"`` uses
    Presidio's predefined pattern recognizers (no spaCy), falling back to regex if unavailable.
    """
    if backend == "presidio":
        return _detect_presidio(text)
    if backend == "regex":
        return _detect_regex(text)
    raise ValueError(f"unknown pattern backend {backend!r}")
