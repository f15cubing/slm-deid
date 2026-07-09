"""End-to-end de-identification pipeline (productionization layer).

This package is the *surrounding pipeline* the BrainLift describes: the model's job is the
context-sensitive **name judgment**; everything else — deterministic pattern PII, applying the
model's judgment without mangling the text, and realistic surrogate replacement — lives here.

Flow (see :mod:`pipeline.deid`)::

    text
      │
      ├─ patterns.detect(text)          # regex email/phone/SSN/… → spans on ORIGINAL coords
      │
      ├─ tagger.tag(text) ──▶ project.project_spans(text, model_tagged)
      │                        # align the model's (possibly drifted) output onto ORIGINAL text
      │                        # by char offset → name spans on ORIGINAL coords (integrity holds)
      │
      └─ merge + render(mode)           # tag | mask | surrogate

Deliberately kept separate from ``src/`` (model, datagen, eval): Colab runs are purely
train/eval, and this layer only *consumes* a trained :class:`src.infer.Tagger`. It never feeds
data back into training, so it cannot leak the eval set.

Tag syntax is imported from :mod:`src.common.tags` — the single source of truth — never
hard-coded here.
"""

from __future__ import annotations

from pipeline.deid import Deidentifier, DeidResult
from pipeline.patterns import PatternSpan, detect
from pipeline.project import ProjectingTagger, project_spans, project_tags
from pipeline.surrogate import SurrogateMap

__all__ = [
    "DeidResult",
    "Deidentifier",
    "PatternSpan",
    "ProjectingTagger",
    "SurrogateMap",
    "detect",
    "project_spans",
    "project_tags",
]
