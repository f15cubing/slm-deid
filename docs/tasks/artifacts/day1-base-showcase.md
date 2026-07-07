# Day 1 — prompted base behavior on the ambiguous showcase (evidence)

Raw outputs of `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` **prompted** (no fine-tune) on the Day-1
showcase, captured live on Colab (T4), greedy decoding, non-thinking mode. This is the qualitative
"before" for SPOV 7 — not scored (that's the Day-2 harness), just the motivating signal.

| # | Intended judgment | Input | Base output | Result |
|---|---|---|---|---|
| 1 | Newton = person → tag | `Newton was frustrated when the experiment failed.` | `⟨NAME⟩Newton⟨/NAME⟩ was frustrated when the experiment failed.` | ✅ correct |
| 2 | Newton method → don't tag | `We applied the Newton method to approximate the root.` | `⟨NAME⟩We⟨/NAME⟩ applied the Newton method to approximate the root.` | ❌ over-tag, wrong span |
| 3 | Newton's laws → don't tag | `Newton's laws describe classical motion.` | `⟨NAME⟩...⟨/NAME⟩` | ❌ degenerate / parroted prompt; integrity fail |
| 4 | Chelsea = place → don't tag | `I visited Chelsea last summer and loved it.` | `⟨NAME⟩Chelsea⟨/NAME⟩` | ❌ over-tag + dropped rest of sentence (integrity fail) |
| 5 | Chelsea = person → tag | `Chelsea helped me revise my thesis statement.` | `⟨NAME⟩Chelsea⟨/NAME⟩ helped me revise my thesis statement.` | ✅ correct |
| 6 | first-name-only → tag Sam | `thanks, Sam — that explanation finally clicked` | `⟨NAME⟩Sam⟨/NAME⟩ — that explanation finally clicked` | ⚠️ tagged Sam but dropped "thanks, " (integrity fail) |
| 7 | tag Rivera, not Grace | `My teacher Ms. Rivera said Grace is a virtue worth practicing.` | `⟨NAME⟩Grace⟨/NAME⟩ is a virtue worth practicing.` | ❌ wrong word tagged + dropped preamble (integrity fail) |

## Read
- **2/7 correct** on the ambiguous set (the two "obvious person" cases). The base is fine on easy
  names and unreliable on everything hard — exactly the gap the fine-tune targets.
- **Two failure families:** (a) judgment errors — over-tagging eponyms/places/common words; (b)
  **integrity violations** — the base silently drops input text (cases 3, 4, 6, 7). `unwrap(output)
  == input` is a hard reject condition, so several of these are automatic FAILs regardless of tags.
- This qualitatively confirms the falsifiable bet's premise. The quantitative base-vs-tuned numbers
  come from the Day-2 harness on the quarantined hard-cases set.

_Note: greedy/1-shot; the point is the pattern of unreliability, not any single sample._
