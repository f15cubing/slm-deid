# De-Id SLM — Context-Sensitive Personal-Name Judgment

Fine-tune **Qwen3-1.7B (QLoRA)** to do the one part of de-identification a prompt can't reliably do:
**context-sensitive personal-name judgment** — deciding, in context, whether a token is a person's
name or an identically-spelled non-person, every time, without drifting.

Pattern types (email / phone / ID) and output format are handled by regex + constrained decoding in
the surrounding pipeline. The **trained behavior, the dataset, and the eval are all about the
judgment core**.

## The behavior (the gate)

> Given a passage of educational text (student essay or tutoring-chat turn), the model returns it
> **unchanged except** that every span referring to a real person's name is wrapped in
> `⟨NAME⟩…⟨/NAME⟩`, and **no other span is tagged**.

- **PASS** = all and only personal-name spans tagged; text otherwise byte-identical to input.
- **FAIL** = any missed name, any non-name tagged, or any other change to the text.

The hard cases (where fine-tune beats prompting): person-vs-eponym (`Newton`), person-vs-place
(`Chelsea`), person-vs-common-word (`Grace`, `Bishop`), first-name-only, possessives, third parties,
and negative traps.

## Repo map

| Path | What |
|---|---|
| `docs/plan.md` | The one-week build plan (Mon Jul 6 → Sun Jul 12). |
| `docs/STATUS.md` | Live "what's done / in-flight / next". Updated on every merge. |
| `docs/agent-workflow-starter-kit.md` | The multi-agent workflow reference this repo's process is built on. |
| `AGENTS.md` | Always-loaded agent rules: hard ceilings + skill routing. |
| `.cursor/skills/` | Project skills: `shipping-changes`, `building-and-testing`. |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See `docs/plan.md` for the day-by-day arc and `AGENTS.md` for how to ship changes here.
