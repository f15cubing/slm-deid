"""End-to-end de-identification orchestrator.

Wires the three stages into one call: deterministic pattern detection, model name judgment
projected onto the original text, then a merge + render step. Render modes:

- ``tag``       — inline markers, ``⟨NAME⟩…⟨/NAME⟩`` / ``⟨EMAIL⟩…⟨/EMAIL⟩`` etc.
                  Integrity-preserving: strips back to the original byte-for-byte.
- ``mask``      — replace each span with a ``[LABEL]`` placeholder.
- ``surrogate`` — replace each span with a consistent, realistic fake (see
                  :class:`pipeline.surrogate.SurrogateMap`).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline import patterns, project
from pipeline.surrogate import SurrogateMap
from src.common.tags import TaggedSpan
from src.infer import Tagger

RENDER_MODES = ("tag", "mask", "surrogate")


@dataclass(frozen=True)
class Entity:
    """A resolved PII span to act on: a projected NAME or a detected pattern type."""

    start: int
    end: int
    text: str
    label: str  # NAME | EMAIL | PHONE | SSN | CREDIT_CARD | IP | URL | ID


@dataclass(frozen=True)
class DeidResult:
    original: str
    text: str  # the rendered output for the chosen mode
    entities: list[Entity]  # everything acted on, sorted by position
    dropped_spans: int  # model name spans that failed to align (see project.project)

    @property
    def name_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.label == "NAME"]

    @property
    def pattern_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.label != "NAME"]


def _tag_markers(label: str) -> tuple[str, str]:
    return (f"⟨{label}⟩", f"⟨/{label}⟩")


def _resolve(
    name_spans: list[TaggedSpan],
    pattern_spans: list[patterns.PatternSpan],
) -> list[Entity]:
    """Combine names + patterns into one non-overlapping, position-sorted entity list.

    Pattern spans are deterministic and high-precision, so on a name/pattern overlap the pattern
    wins (the name span is dropped). Within each source the spans are already non-overlapping.
    """
    names = [Entity(s.start, s.end, s.text, "NAME") for s in name_spans]
    patt = [Entity(p.start, p.end, p.text, p.label) for p in pattern_spans]

    # priority 0 = pattern (kept first on conflict), 1 = name
    ordered = sorted(
        [(0, e) for e in patt] + [(1, e) for e in names],
        key=lambda pe: (pe[1].start, pe[0]),
    )
    kept: list[Entity] = []
    for _prio, e in ordered:
        if any(e.start < k.end and k.start < e.end for k in kept):
            continue
        kept.append(e)
    return sorted(kept, key=lambda e: e.start)


def _render(original: str, entities: list[Entity], mode: str, surrogates: SurrogateMap) -> str:
    out: list[str] = []
    cursor = 0
    for e in entities:
        out.append(original[cursor : e.start])
        if mode == "tag":
            open_m, close_m = _tag_markers(e.label)
            out.append(f"{open_m}{e.text}{close_m}")
        elif mode == "mask":
            out.append(f"[{e.label}]")
        elif mode == "surrogate":
            out.append(surrogates.get(e.label, e.text))
        else:  # pragma: no cover - guarded by deidentify
            raise ValueError(f"unknown render mode {mode!r}")
        cursor = e.end
    out.append(original[cursor:])
    return "".join(out)


class Deidentifier:
    """Compose a pattern detector, a name :class:`~src.infer.Tagger`, and offset projection.

    The tagger is injected, so the same pipeline serves the base model, the tuned adapter, or a
    stub — and unit tests run with no GPU. ``pattern_backend`` and ``surrogate_seed`` are
    plumbed through to :mod:`pipeline.patterns` and :class:`SurrogateMap`.
    """

    def __init__(
        self,
        tagger: Tagger,
        pattern_backend: str = "regex",
        surrogate_seed: int = 0,
    ):
        self.tagger = tagger
        self.pattern_backend = pattern_backend
        self.surrogate_seed = surrogate_seed

    def deidentify(self, text: str, mode: str = "tag") -> DeidResult:
        if mode not in RENDER_MODES:
            raise ValueError(f"mode must be one of {RENDER_MODES}, got {mode!r}")

        model_tagged = self.tagger.tag(text)
        proj = project.project(text, model_tagged)
        pattern_spans = patterns.detect(text, backend=self.pattern_backend)

        entities = _resolve(proj.spans, pattern_spans)
        surrogates = SurrogateMap(seed=self.surrogate_seed)
        rendered = _render(text, entities, mode, surrogates)
        return DeidResult(
            original=text,
            text=rendered,
            entities=entities,
            dropped_spans=proj.dropped,
        )
