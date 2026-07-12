# BrainLift v3 — De-Id SLM (Source of Truth)

> Structured context for AI conversations. **v3 re-centers the SLM's mandate** on the one part of de-identification that neither a rule nor a decoder can do — context-sensitive name/address judgment — resolving the v2 worry that "the fine-tune is the least important piece." Built bottom-up: facts (DOK 1) → organized knowledge (DOK 2) → insights (DOK 3) → spiky POVs (DOK 4). Material from the Avenue 1–3 briefs is marked **[v2]**; the scope re-centering is marked **[v3]**. All re-identification figures are from the literature and kept strictly separate from any not-yet-measured risk on our own corpus.

---

## Owners

- [Felipe]
- [Name 2]

---

## Purpose

**Purpose.** Establish what it actually takes to make a small, local, open-weight model reliably perform one narrow, safety-critical behavior — de-identifying PII in educational text — and develop a defensible point of view on why reliability engineered through data beats capability accessed through prompting. Core belief: the hard, valuable part of applied LLM work is controlled, verifiable behavior on the long tail, not model intelligence.

**The model's mandate [v3].** The SLM's job is the one part of de-identification that neither a rule nor a decoder can do: **context-sensitive judgment** — deciding which spans are names/addresses *in context* (is "Newton" a person or a method? is "Chelsea" a person, a place, or a team?), and optionally generating coherent same-type surrogates. Everything with a surface form — emails, phones, SSNs, dates, IDs, URLs, and output formatting — is explicitly **not** the model's job; it belongs to regex/checksums and constrained decoding. The model spends 100% of its learned capacity on the ambiguous residue.

**In Scope.**

- **[v3]** The SLM's core: context-sensitive PII **judgment** on ambiguous spans (names/addresses), and optionally coherent surrogate **generation** — the un-regexable, un-decodable slice.
- **[v3]** The **dataset that teaches that judgment** (the actual deliverable), plus a before/after **vs. prompting** on exactly the hard, ambiguous cases (the assignment's litmus comparison).
- Original, defensible positions on de-id as an engineering problem (scope of claim, schema, data strategy, evaluation).
- **[v2]** The supporting pipeline — regex/Presidio for pattern types + grammar-constrained decoding for format + a verifier, with inline typed tagging as the schema — used as scaffolding **around** the model, not as the model's task.
- **[v2]** A recall-weighted, **entity-level** evaluation and an honest CI-bounded pilot.

**Out of Scope.**

- **[v3]** Treating pattern detection or output formatting as the model's job (regex and constrained decoding own those).
- Using AI to invent the insights; pretraining from scratch; RLHF; RAG retrieval infra.
- **[v2]** In-text quasi-identifier masking (moved to metadata SDC + access governance).
- **[v2]** Claiming "anonymization"; full generative rewriting; unverified span-offset output.

---

## Main Conclusions (from the Avenue 1–3 research, re-scoped in **[v3]**)

**Scope re-centering [v3].** v2 concluded the fine-tune was "the smallest, most replaceable piece." That is true of the *production pipeline* but was the wrong unit of analysis. The assignment's unit is *an SLM reliably doing one thing a prompt can't*, and its litmus test is **fine-tune vs. prompt** — **not** fine-tune vs. regex+constrained-decoding (the comparison Avenue 3's 15-pt bar actually made, which is a production question). Re-scoped to the one part of de-id that is genuinely un-regexable and un-decodable — **context-sensitive name/address judgment** — the fine-tune is not the least important piece; it is the point. Regex can't do it (names aren't a pattern), constrained decoding can't (it only enforces format, detects nothing), prompting can but unreliably (precision ~0.24, over-flags "the Newton method," drifts after a few samples), and fine-tuning beats prompting on it (NAME recall 0.78→0.96, Shen et al.). So the hard sub-task passes the assignment's actual litmus test cleanly; only the surrounding easy work doesn't need the model.

Three design questions remain resolved as below.

1. **Scope & claim (Avenue 1 → COMBINE).** Scope the shipped privacy claim of the Qwen3-1.7B redactor to **direct identifiers only**. Push quasi-identifiers **downstream** to tabular statistical disclosure control (SDC) on the metadata (small-cell suppression <5, aggregation, generalization, noise, rounding) + an **Expert-Determination-style risk assessment + a Data Use Agreement**. Label output "de-identified: direct identifiers removed," **never "anonymized."** *Flip only if* a pilot shows a CI-separable, non-trivial share of documents re-identifiable **from the prose itself** against plausible auxiliary data, **or** release changes from DUA-gated to open public.
2. **Training data (Avenue 2 → COMBINE / mixed).** Use a **mixed strategy**: the Presidio Sentence Faker as the free-exact-span labeling backbone + LLM-distilled realistic carriers + entity-swap augmentation of real text + **a small real slice**. Synthetic gets you *most* of the way; a small real slice is likely **decisive** for the last several NAME points. The residual gap concentrates almost entirely on **NAME** (pattern types transfer via regex). *Flip to mandatory real data* if the synthetic-only vs. synthetic+small-real **NAME F5 gap exceeds ~5 points** with separable CIs (target ~1:1 mix, per Enache et al.).
3. **Architecture (Avenue 3 → ADOPT hybrid, re-read in v3).** Ship a **hybrid**: deterministic regex/Presidio for pattern types + **grammar-constrained decoding** for format + a **fine-tuned LM for the NAME/ADDRESS judgment core** + a **verifier pass**, with **inline typed tagging** as the schema. *(v3 re-read: the fine-tune's job is the judgment core, benchmarked against* ***prompting*** *— the assignment's test. Avenue 3's ≥15 pp bar was the fine-tune-vs-regex+CD production comparison; keep it as a* ***pipeline-composition*** *check, not as the test of whether the project is worthwhile.)*

**Overarching thesis [revised v3]:** For the *production pipeline*, the fine-tune is one component among several. For the *assignment* — and for the one genuinely hard part of de-identification — it is the **load-bearing** piece, because context-sensitive judgment is exactly what rules and decoders cannot do and what prompting cannot do reliably. Scope the model to that core; make the **dataset that teaches the judgment** the deliverable; evaluate it **against prompting on the hard, ambiguous cases**, with recall-weighted, entity-level metrics and an honest CI-bounded pilot. Recall (entity-level, F5-weighted) governs the *detector*; residual re-identifiability governs the *privacy claim*.

---

## DOK 1 — Facts

*(F1–F27 unchanged from v1; abbreviated. F28–F50 are new* ***[v2]****, sourced to the Avenue 1–3 briefs.)*

**Prior facts (v1), condensed.** F1–F2: fine-tuned small model hit recall 0.9589 on CRAPII essays / 0.9895 on TSCC tutoring chat, ~3× precision over prompting (Shen et al.). F3: GPT-NER inline-tag + regex-extract avoids offset hallucination/over-labeling. F4: prompted LLMs get high recall, low precision (one system 0.24). F5–F7: HIPS surrogate replacement; humans recover only ~27% of missed PII; BRATsynthetic cut doc-level leakage 94.2%→57.7% at 5% FN. F8–F9: CRAPII (22,688 essays, 7 types) / TSCC (260 sessions). F10: Learning Agency Lab scored micro **F-beta β=5**. F11: public educational PII data heavily LLM-generated. F12–F14: n2c2-2014 (1,304 notes, HIPAA-18) needs a DUA; MIMIC needs credentialing. F15–F17: ai4privacy 200k/300k; TAB (ECHR, direct+quasi). F18–F22: Presidio (regex+spaCy); Sentence Faker; consistent surrogates; Faker for patterned fields. F23–F24: generative rewriting alters content; small-model surrogate pipelines have honest-negative results. F25–F27: Qwen3-1.7B (Apache-2.0, thinking toggle); QLoRA fits one 24GB GPU, minutes to train.

**Regulatory & re-identification science [v2]**

- F28. HIPAA offers two de-id pathways: **Safe Harbor** (remove 18 identifiers + no actual knowledge remaining data identifies) and **Expert Determination** (qualified expert, generally accepted statistical principles, documents "very small" residual risk). Neither removes all risk. (HHS OCR · regulatory · high)
- F29. Safe Harbor also requires stripping dates finer than year, geography finer than the first 3 ZIP digits, and ages >89 — so a direct-identifier-only text redactor matches Safe Harbor's *mechanics* but not its full checklist, and does **not** by itself satisfy Expert Determination (an in-context risk assessment, not a redaction technique). (HHS OCR · regulatory · high)
- F30. Sweeney (2002): **87%** of the 1990 US population (216M/248M) were likely unique on {5-digit ZIP, gender, DOB}. (peer-reviewed · high)
- F31. Rocher, Hendrickx & de Montjoye (2019, *Nature Communications*): a generative model estimated **99.98%** of Americans re-identifiable from 15 demographic attributes (AUC 0.84–0.97 across 210 populations). (peer-reviewed · high)
- F32. TAB (1,268 ECHR cases, 155,006 mentions): only **4.4%** of annotated entities were direct identifiers, **63%** quasi-identifiers, **32%** left unmasked; an entity is "protected" only if **all** its mentions are masked; quasi-identifier categories are **"unbounded."** (benchmark · high)
- F33. TAB: mention-level recall ~0.7 corresponded to direct-identifier **entity-level** recall ~0.45 — a method can look good on recall while leaking identities. (benchmark · high)
- F34. Yacobson et al. (2021, *JLA*): from de-identified ITS interaction data (~600 fifth-graders, 16 classes, 5 schools) linked to public web data, class membership was re-identified at **adjusted Rand index 0.943**; "de-identification alone is insufficient." (peer-reviewed · high)
- F35. MathEd-PII (Zhou et al., 2026, EDM): naive redaction of math-tutoring dialogue over-redacted — **64.87%** of the provider's redactions were "Not PII," ~4.88:1 false-to-true in math-dense segments — while still **missing 159 real names** Presidio failed to catch. (preprint · medium)
- F36. Narayanan et al. (2012, IEEE S&P): stylometry on 100,000 blog authors identified the author in **>20%** of cases, top-20 in **~35%**; small candidate pools exceed 90%. (peer-reviewed · high)
- F37. Education records are typically governed by **FERPA** (COPPA for under-13s), not HIPAA; FERPA mirrors HIPAA's two methods and adds a **"reasonable certainty"** standard explicitly covering indirect identifiers. (regulatory · high)
- F38. 3ie SDC methods for the metadata layer — small-cell suppression (<5), aggregation, generalization, noise, rounding — offer measurable **k-anonymity** guarantees no text model can provide. (practitioner · high)

**Synthetic data & the sim-to-real gap [v2]**

- F39. Enache et al. (2025, colonoscopy NER): synthetic-trained **F1 0.99 in-domain but 0.33 on real**; a **1:1 real:synthetic mix recovered to 0.94** (vs real-only 0.70, synthetic-only 0.64, p<0.001) — "a valuable complement, but not a substitute for real annotations." (peer-reviewed · med-high)
- F40. Presidio Sentence Faker returns **exact character spans** for injected entities (free labels) and provides **leakage-safe splitting** (a template appears in only one fold). (tool docs · high)
- F41. Dai & Adel (COLING 2020): simple NER augmentation gives its largest gains in low-resource regimes (up to **~+9.6 F1 at 50 sentences**) and shrinks to ~zero (occasionally negative) at full data. (peer-reviewed · high)
- F42. CuratorKIT's PIIPseudonymizer replaces PII with realistic same-type surrogates (not `[PERSON]`) via a per-sample entity map + seeded Faker, because generic placeholders corrupt structure and degrade downstream generation. (tool/paper · medium)
- F43. Halterman (*Political Analysis* 2025): a marginal hand-annotated synthetic example is **less useful than a real one**, and synthetic augmentation + 100–500 hand-labeled examples beats 1,000 hand-annotated documents alone — real labels carry disproportionate weight. (peer-reviewed · high)
- F44. CRAPII: 22,688 files, **31.2%** contain ≥1 PII across 14 types, HIPS surrogate obfuscation. TSCC: 260 lessons, 41.4K turns, 363K tokens, 2 teachers/13 students. (benchmark · high)
- F45. Carrell et al. (2013, *JAMIA*): HIPS surrogates concealed ~90% of missed identifiers, but a later **"parrot attack" re-exposed ~two-thirds** of leaked PII — camouflage is strong but not absolute. (peer-reviewed · high)

**Architecture, constrained decoding & the fine-tuning question [v2]**

- F46. CMU educational de-id study: a **verifier pass lifted NAME precision 0.6109 → 0.8830 but cut recall 0.9605 → 0.8038**; fine-tuning did **not** beat prompting on overall precision (0.6042 vs 0.6593) — its win was **recall** (0.7781 → 0.9589 overall). (preprint · medium) *(Attribution note: the 0.61→0.88 verifier figure is from THIS study, not from the GPT-NER paper.)*
- F47. Grammar-constrained decoding (XGrammar-2) raised Llama-3.2-1B tool-call JSON schema-validity **22.07% → 100.00%**; SLOT reports Llama-3.2-1B schema accuracy **88.9% (SFT alone) → 96.2% (SFT+XGrammar)** — fine-tuning and constrained decoding are **complementary, not competing**. (preprint · reported · med-high)
- F48. Tam et al. ("Let Me Speak Freely?", EMNLP 2024): stricter output-format constraints **degrade LLM reasoning**; a format guarantee is **not** a semantic guarantee. (peer-reviewed · high)
- F49. Clinical de-id ceilings (reference bar): top i2b2-2014 F1 **.936** (Stubbs et al. 2015); later ensembles **~0.985** (0.979P/0.992R); Philter **99.92% recall** / F2 94.77% (Norgeot 2020). (peer-reviewed · high)
- F50. **Because constrained decoding already guarantees format and regex/checksums already catch pattern-type PII (email/phone/SSN/date/ID), the only non-redundant contribution of fine-tuning is context-dependent NAME/ADDRESS precision.** (synthesis · high)
- F51. **[v3]** Structural: regex cannot detect names (they are not a regular pattern), and grammar-constrained decoding only constrains output *format* and performs no detection — so neither can do context-sensitive entity detection; **only a learned model or prompting can**, and prompting does so unreliably (F4, F46). Fine-tuning beats prompting on exactly this sub-task (overall NAME recall 0.78→0.96, Shen et al.). (synthesis · high)

---

## DOK 2 — Knowledge Tree

*(Categories 1–4 from v1 retained. Categories 5–7 are new* ***[v2]****.)*

- **Category 1–4 (v1):** the direct-precedent educational de-id study (S1); educational/clinical/general/legal datasets (CRAPII, TSCC, n2c2, ai4privacy, TAB); Presidio + generation tooling; Qwen3-1.7B + QLoRA.

**Category 5 — Regulatory standards & re-identification science [v2]**

- *HHS OCR De-identification Guidance* (Safe Harbor / Expert Determination; 45 CFR 164.514) — F28, F29. *Summary:* defines the two lawful bars; a text redactor satisfies neither by itself — the claim must be scoped and governed.
- *US Dept. of Education / FERPA* (studentprivacy.ed.gov) — F37. *Summary:* the operative regime for student text; "reasonable certainty" covers indirect identifiers.
- *Sweeney 2002; Rocher et al. 2019; Narayanan et al. 2012; Yacobson et al. 2021; 3ie 2025* — F30, F31, F36, F34, F38. *Summary:* residual re-id risk from quasi-identifiers/stylometry is large in the abstract and demonstrated in education; the metadata tuple is the tractable control point (SDC → k-anonymity).

**Category 6 — Synthetic data generation & the sim-to-real gap [v2]**

- *Enache et al. 2025; Halterman 2025; Dai & Adel 2020* — F39, F43, F41. *Summary:* pure synthetic collapses on real (0.99→0.33), a mix recovers (→0.94), real labels carry outsized weight; augmentation helps most when data is scarce. Synthetic = backbone, not substitute.
- *Presidio Sentence Faker; CuratorKIT; Carrell et al. 2013* — F40, F42, F45. *Summary:* free exact spans + leakage-safe splits; consistent realistic surrogates preserve utility; surrogate camouflage is strong but re-attackable.

**Category 7 — Output schema, constrained decoding & the fine-tuning question [v2]**

- *GPT-NER; XGrammar/XGrammar-2; SLOT; Tam et al. 2024; Anonpsy* — F46, F47, F48. *Summary:* inline tagging is checkable; constrained decoding guarantees format (not semantics) and can hurt reasoning; fine-tuning + decoding are complementary.
- *CMU verifier result; i2b2 ceilings (Stubbs 2015, Norgeot 2020)* — F46, F49. *Summary:* the one clean fine-tuning NAME-precision win is recall-costly; strong clinical systems set a high recall bar.

---

## DOK 3 — Insights

*(Insights 1–8 from v1 retained; abbreviated. 9–13 are* ***[v2]****; 14–15 are new* ***[v3]****.)*

**Prior insights (v1), condensed.** (1) Difficulty is entirely in the long tail — the per-instance reliability a prompt can't guarantee. (2) Surrogate replacement changes what "success" means (exploitable leakage, not raw recall). (3) The output schema is a risk-control decision, not formatting. (4) Public educational data is a trap if used naively. (5) Category-based de-id ≠ true anonymization. (6) The privacy rationale and the technical choice are the same choice. (7) A hybrid lets the model spend scarce capacity where it matters. (8) Precision is a safety metric in disguise under HIPS.

- **Insight 9 [v2] — The real leak vector is the metadata, not the prose.** The re-identifying tuple (age + role + location + household) lives in structured fields where uniqueness is computable and controllable by SDC; TAB shows quasi-identifiers are 63% of text entities and "unbounded," so a text model can never certify k-anonymity. The SLM should own direct identifiers in text; metadata SDC + governance should own quasi-identifiers. *Built from F32, F38, F30, F31.*
- **Insight 10 [v2] — Mention-level recall is a vanity metric.** The honest unit is the *entity* (all mentions masked), and above that, *residual re-identifiability*. TAB's 0.7→0.45 mention-vs-entity gap shows a detector can look safe while leaking; even entity-level direct-identifier recall doesn't bound quasi-identifier or stylometric re-id. *Built from F33, F34, F36.*
- **Insight 11 [v2] — Clean synthetic data manufactures false confidence; a small real slice outweighs a large synthetic one.** Enache's 0.99→0.33 collapse (and 1:1 recovery to 0.94) plus Halterman's real-label weighting mean synthetic is the coverage backbone but the last NAME points need real data — and only a *messy real held-out* set can detect the gap. *Built from F39, F43, F45.*
- **Insight 12 [v2] — Fine-tuning's value is narrow, recall-costly, and possibly dominated by simpler layers.** Constrained decoding guarantees format; regex catches pattern PII; so fine-tuning's only live contribution is NAME/ADDRESS precision — and the one clean measurement shows that precision gain trades away recall, which under F5 weighting may not pay. *Built from F46, F47, F50.*
- **Insight 13 [v2] — Format reliability and content correctness are independent problems.** Constrained decoding gives a format guarantee, not a semantic one, and over-constraining can degrade judgment — so the pipeline needs *both* a format layer (constrained decoding) and a correctness check (verifier), not one instead of the other. *Built from F47, F48.*
- **Insight 14 [v3] — The litmus test is fine-tune vs.** ***prompt*****, not fine-tune vs.** ***regex+constrained-decoding*****; conflating them is what made the fine-tune look dispensable.** Avenue 3's 15-pt bar compared the model to a rules+decoder baseline — a production-composition question. The assignment asks a different question — can a *dataset* make the model reliably do what a *prompt* can't? — and on the sub-task that matters (names-in-context), the answer is yes. *Built from F4, F46, F50, F51.*
- **Insight 15 [v3] — Scope the model to the un-regexable core, and both the tension and the over-redaction risk dissolve.** The model's capacity should go entirely to context-sensitive judgment (name vs. Newton-the-method; person vs. place) and, optionally, coherent surrogate generation — itself inherently generative and un-promptable-reliably (on-device regurgitation findings). Pattern types and format go to the pipeline, which is where over-redaction of things like math tokens is best prevented. *Built from F7, F24, F35, F50, F51.*

---

## DOK 4 — Spiky Points of View

*(SPOV 1, 4, 6 retained from v1. SPOV 2, 3, 5 updated with Avenue-1–3 evidence. SPOV 7* ***revised in [v3]*** *— from "fine-tuning may not earn its keep" to "it earns its keep exactly on the judgment core." SPOV 8 is new* ***[v3]****.)*

- **SPOV 1 [nuanced v3] — The** ***deliverable*** **is the dataset and the eval harness, not the model artifact — but "the dataset is the deliverable" does not mean "the fine-tune is dispensable."** The thing you hand in and the thing that does the hard work are different: you're graded on the dataset that teaches context-sensitive judgment (and the before/after-vs-prompting eval), while the fine-tune trained on it is what actually performs the un-regexable core. v2 briefly over-rotated this SPOV into "the fine-tune is the least important component"; v3 corrects that (see SPOV 7/8) — at the *pipeline* level the fine-tune is small, but for the *task* it is load-bearing. At small n the deliverable is the protocol + a CI-bounded pilot, not a significant effect. *Facts: F1, F25, F27, F50, F51; Main Conclusions.*
- **SPOV 2 [updated v2] — Your headline metric is lying to you: mention-level recall and F1 are vanity numbers. The only honest safety metrics are entity-level leakage (every mention masked) and residual re-identifiability.** Raw recall/F1 hide the gap (TAB: 0.7 mention → 0.45 entity); surrogate replacement means what matters is *exploitable* leakage; and even perfect direct-identifier entity-recall doesn't bound quasi-identifier or stylometric re-id. Recall still governs the detector (F5-weighted, entity-level) — but as a floor, not a trophy. *Facts: F5–F7, F10, F33, F34, F36. Insights 2, 10.*
- **SPOV 3 [updated v2] — Clean synthetic data is a liability you must actively distrust; the more impressive your in-distribution number, the more suspicious you should be.** Pure synthetic collapses on real text (0.99→0.33) and recovers only with a real slice (→0.94); a marginal real label beats a marginal synthetic one. So synthetic is *training-breadth only*, a small real slice is decisive, and you should deliberately build an *ugly* held-out set to make your model look worse than the headline. *Facts: F11, F39, F43, F45. Insights 4, 11.*
- **SPOV 4 [retained, nuanced v2] — Inline tagging is the only responsible schema for a generative de-identifier, because it's the only one you can mechanically prove didn't lie — but format-provable is not content-correct.** Emit inline tags under grammar-constrained decoding (format guaranteed) *and* diff output-minus-tags against the source *and* run a verifier — because a format guarantee is not a semantic guarantee, and over-constraining can degrade judgment. Rewriting and unverified offsets are disqualified. *Facts: F3, F23, F47, F48. Insights 3, 13.*
- **SPOV 5 [updated v2] — Scope the claim to direct identifiers, move quasi-identifiers to metadata SDC + a DUA, and say so out loud — because the metadata is the real threat and calling direct-only redaction "anonymization" is precisely what FERPA/HIPAA forbid.** The re-identifying tuple lives in structured fields a text model can't reach; forcing quasi-masking into a 1.7B model over-redacts (MathEd-PII: 64.87% false redactions) and destroys the utility that motivates sharing. Ship the quasi-identifier limitation as an explicit, demonstrated caveat, not a buried footnote. *Facts: F28–F32, F35, F37, F38. Insights 5, 9.*
- **SPOV 6 — The small local model is the correct tool on the merits, not a privacy-forced downgrade.** The data can't leave the machine (DUAs, credentialing), so a local open-weight model isn't a weaker substitute for the "real" tool — it *is* the only admissible tool; the frontier model's capability is irrelevant when it can't be legally used. *Facts: F14, F37, F25, F27. Insight 6.*
- **SPOV 7 [revised v3] — Fine-tuning earns its keep exactly where nothing else can, and nowhere else.** Benchmark it against the wrong baseline (regex + constrained decoding) and it looks dispensable; benchmark it against the assignment's baseline — **prompting** — on context-sensitive name/address judgment and it is load-bearing. The v2 "least important piece" read was a **scope error**: measuring the production pipeline, not the task. Constrained decoding and regex own format and pattern types; the model owns the one thing they *structurally cannot do*. **Falsifier (unchanged in spirit):** if a pilot shows fine-tuning does **not** beat prompting on the hard, ambiguous cases (recall/precision on names-in-context), that is a real signal to pivot — the whole re-centering is a falsifiable bet that it does. *Facts: F4, F46, F50, F51. Insights 12, 14, 15.*
- **SPOV 8 [new v3] — The model's job is judgment, not detection-of-patterns or formatting — and that reframing turns "the dataset is the deliverable" from a slogan into the literal project.** De-identification's only un-automatable core is deciding, *in context*, whether a token is a person / place / thing. Regex, checksums, and grammars handle everything with a surface form; the SLM should spend 100% of its learned capacity on the ambiguous residue. The deliverable is therefore the **dataset of hard, contextually-ambiguous judgment calls** (and their surrogates), and success is measured **against prompting on exactly those cases** — not against a pipeline that was never trying to do the hard part. This is the assignment's own philosophy taken literally. *Facts: F35, F50, F51. Insights 14, 15.*

---

## Empirical verdict [v3] — did data→behavior hold?

*Added 2026-07-12, closing the loop the plan required ("BrainLift — whether data→behavior held, with
evidence"). Resolves SPOV 7's falsifiable bet. Numbers are the canonical live-teacher run
(`sft-v3-gpt551`) on the 51 quarantined hard cases; full tables + 95% bootstrap CIs in
[`results.md`](results.md) → gpt551 and [`final-report.md`](final-report.md).*

**The bet (SPOV 7 falsifier).** *If a fine-tune does not beat prompting on the hard, ambiguous
name-in-context cases, pivot.* This was a real falsifiable claim, not decoration.

**Verdict: the bet HELD — data→behavior held, not refuted.** Fine-tuning the same 1.7B model on the
judgment dataset beat the prompted base on **every** axis, with CIs separated on the recall-family
metrics:

| | Prompted base | Fine-tuned (gpt551) |
|---|---|---|
| F5 / recall | 0.51 / 0.52 | **0.85 / 0.85** |
| pass rate | 0.35 | **0.82** |
| over-tag ↓ | 0.55 | **0.16** |
| integrity violation ↓ | 0.59 | **0.02** |
| leakage ↓ | 0.26 | **0.08** |

- **It's judgment, not memorization (SPOV 8 confirmed).** On names **never seen in training**
  (held-out-names probe) base→tuned recall went 0.08 → 1.00; on a surface-disjoint OOD probe,
  0.05 → 0.89. The behavior transferred to novel tokens — the model learned the *decision*, not the
  vocabulary. This is the strongest evidence that the **dataset** (SPOV 1) is what did the work.
- **Right tool on the merits (SPOV 6), quantified.** A frontier model (gpt-4.1) scored pass 0.88 on
  the same hard cases — the 1.7B local tune (0.82) is *competitive* and actually **beats the frontier
  on recall** (gpt-4.1 is precision-first and under-recalls). A model that can be run locally on
  credentialed data is within striking distance of one that legally cannot be. See
  [`eval-engine-comparison.md`](eval-engine-comparison.md).
- **The leakage ceiling held (SPOV 2/4).** 0 exact + 0 substring overlap between the 818/90 training
  splits and all 201 quarantined eval inputs, independently re-verified; enforced in CI.

**Honest limits — reported, not hidden.**
1. **Small evidence base.** n=51 hard cases, single seed, single teacher + single verifier pass. The
   deliverable is the *protocol + a CI-bounded pilot*, not a significant effect (SPOV 1's own caveat).
2. **gpt551 scores below the earlier authored run** (pass 0.82 vs 0.96). We do **not** claim it beats
   it — the most likely reason is distributional (authored templates sit closer to the eval), which
   makes the live-teacher number the more *credible* estimate, not the highest one.
3. **`possessive` is the unmoved category** and carries the single residual integrity violation — the
   clearest next data-iteration target (a data fix, per the Day-4 rule, never an HP tweak).
4. **Byte-identity on messy real text still fails** under the "regenerate the passage verbatim" output
   format (SPOV 4's format-vs-content gap made concrete); the pipeline's tag-by-offset projection is
   the structural fix (`unwrap(project(...)) == original` by construction), not more training.

**Bottom line:** on this task, against this baseline, reliability engineered through data beat
capability accessed through prompting — the project's core thesis, measured and held.

---

## Source Index

*Conclusions above derive from three internal research briefs —* ***Avenue 1 (quasi-identifiers), Avenue 2 (synthetic data / sim-to-real), Avenue 3 (architecture & fine-tuning)*** *— plus the v1 evidence base.*

**Peer-reviewed:** Shen/Ji/Lin/Koedinger, *Enhancing De-identification of PII in Educational Data* (arXiv:2501.09765). Pilán et al., *TAB* (Computational Linguistics 48(4), 2022). Sweeney, *k-anonymity* (2002). Rocher, Hendrickx & de Montjoye, *Nature Communications* 10:3069 (2019). Yacobson et al., *JLA* 8(2) (2021). Narayanan et al., *On the Feasibility of Internet-Scale Author Identification* (IEEE S&P 2012). Shero et al., *AERA Open* 11(1) (2025). Enache et al., *Mixed Real–Synthetic NER* (Medicina/PMC12387308, 2025). Halterman, *Synthetically generated text for supervised text analysis* (Political Analysis 2025). Dai & Adel, *Simple Data Augmentation for NER* (COLING 2020). Carrell et al., *Hiding in Plain Sight* (JAMIA 2013). Tam et al., *Let Me Speak Freely?* (EMNLP 2024). Stubbs, Kotfila & Uzuner (JBI 2015); Norgeot et al. (npj Digital Medicine 2020). Lakens, *Equivalence Testing* (2017/2018).

**Regulatory:** HHS OCR *De-identification Guidance* (45 CFR 164.514). US Dept. of Education / FERPA (studentprivacy.ed.gov).

**Preprint / benchmark / tool:** GPT-NER (arXiv:2304.10428). XGrammar / XGrammar-2 (arXiv:2601.04426). SLOT (arXiv:2505.04016). Anonpsy (arXiv:2601.13503). MathEd-PII / Zhou et al. (arXiv:2602.16571, EDM 2026). CuratorKIT (arXiv:2606.21631). Sadani & Kumar (arXiv:2605.13538). ai4privacy pii-masking-200k/300k. CRAPII (Holmes et al., EDM 2024). TSCC (Caines et al., 2020/2022). Microsoft Presidio + presidio-research (Sentence Faker; leakage-safe splitting). Learning Agency Lab PII Data Detection (Kaggle 2024; F-beta β=5). Qwen3 Technical Report (arXiv:2505.09388); QLoRA (arXiv:2305.14314).

*Caveat on recency: several sources carry 2026 arXiv IDs and are preprints/to-appear; treat their specific numbers as directional and verify against the primary source before citing externally. Throughout, keep [measured-on-our-eval] separate from [reported-in-literature].*
