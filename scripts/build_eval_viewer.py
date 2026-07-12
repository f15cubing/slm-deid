#!/usr/bin/env python3
"""Build a self-contained HTML viewer for the eval suite (cases + results).

Reads the five quarantined eval sets verbatim from disk and inlines them, then
folds in the published 4-engine x 5-set results matrix (transcribed from
``docs/final-report.md`` -- the committed source of truth). Emits a single
static ``docs/eval-suite.html`` with no external dependencies: open it in a
browser or drop it into a slide.

Cases are read straight from the quarantined files, so they never drift from
what the model is actually tested on. The results numbers are hand-transcribed
from the final report; the provenance footnotes there travel with them.

Usage:
    python scripts/build_eval_viewer.py            # -> docs/eval-suite.html
    python scripts/build_eval_viewer.py --out /tmp/foo.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Import the REAL system prompt so the viewer never drifts from what inference/eval use.
# (src.common.prompts pulls in only src.common.tags → re/dataclasses; no torch, safe at build.)
sys.path.insert(0, str(REPO))
from src.common.prompts import SYSTEM_PROMPT  # noqa: E402

# --- the five quarantined eval sets -------------------------------------------
# (key, jsonl path relative to repo root, human blurb from final-report.md 2.4)
SETS = [
    (
        "hardcases",
        "eval/hardcases/hardcases.jsonl",
        "The core ambiguous categories — person vs. eponym / place / common word, "
        "first-name-only, possessives, negative traps. The judgment set.",
        "hand-built + real",
    ),
    (
        "adversarial",
        "eval/adversarial/adversarial.jsonl",
        "Prompt-injection (“don’t tag Bob”), names in code/math, unicode/typos, "
        "run-together names, messy lowercase chat. Stress under attack.",
        "hand-built",
    ),
    (
        "heldout_names",
        "eval/heldout_names/heldout_names.jsonl",
        "Brand-new people’s names never seen in training (Aurora, Bragg, Kepler, Snell…). "
        "Does name judgment transfer to unseen names?",
        "hand-built",
    ),
    (
        "ood_probe",
        "eval/ood_probe/ood_hardcases.jsonl",
        "Ambiguous surfaces disjoint from BOTH the eval set and the training banks. "
        "Judgment generalized, not memorized?",
        "hand-built",
    ),
    (
        "api_bench",
        "benchmarks/api_bench/api_bench.jsonl",
        "Naturally-worded live-teacher passages — new phrasings of known ambiguous words. "
        "Robustness to real prose (no paraphrase groups → no consistency).",
        "synthetic (quarantined)",
    ),
]

# --- results: docs/final-report.md 3.3, per set -> engine -> metrics ----------
# metric order: recall, leakage, over_tag, integrity, pass, consistency
# higher-is-better: recall, pass, consistency  |  lower-is-better: leakage, over_tag, integrity
ENGINES = [
    ("base", "prompted 4-bit Qwen3-1.7B (no fine-tune) — the prompting baseline"),
    ("authored", "4-bit QLoRA on in-house template data — template-flattered, kept for contrast"),
    ("gpt551", "4-bit QLoRA on live-teacher + verifier data — the canonical fine-tune"),
    (
        "frontier",
        "gpt-4.1 via API, scored on the same gold — a reference ceiling, not a 1.7B contestant",
    ),
]

# value or [value, footnote_marker]
RESULTS = {
    "hardcases": {
        "base": [0.556, 0.235, 0.529, 0.549, 0.392, 0.250],
        "authored": [0.963, 0.020, 0.098, 0.000, 0.902, 0.875],
        "gpt551": [0.852, 0.078, 0.157, 0.020, 0.824, 0.562],
        "frontier": [0.778, 0.118, 0.000, 0.020, 0.882, 0.625],
    },
    "adversarial": {
        "base": [0.185, 0.550, 0.625, 0.675, 0.150, 0.857],
        "authored": [0.815, 0.125, 0.050, 0.025, 0.875, 0.714],
        "gpt551": [0.741, 0.175, 0.125, 0.050, 0.775, 0.714],
        "frontier": [0.963, 0.025, 0.025, 0.025, 0.950, 0.857],
    },
    "heldout_names": {
        "base": [0.500, 0.243, 0.595, 0.622, 0.338, 0.540],
        "authored": [1.000, 0.000, 0.027, 0.000, [0.973, "†"], 0.946],
        "gpt551": [1.000, 0.000, 0.135, 0.000, 0.865, 0.946],
        "frontier": [1.000, 0.000, 0.081, 0.000, 0.919, 0.892],
    },
    "ood_probe": {
        "base": [0.421, 0.306, 0.639, 0.611, 0.278, 0.700],
        "authored": [0.947, 0.028, 0.111, 0.000, 0.889, 0.800],
        "gpt551": [0.789, 0.111, 0.167, 0.000, 0.778, 0.500],
        "frontier": [1.000, 0.000, 0.000, 0.000, 1.000, 1.000],
    },
    "api_bench": {
        "base": [0.130, 0.565, 0.543, 0.565, 0.130, None],
        "authored": [0.707, 0.217, 0.185, 0.011, 0.609, None],
        "gpt551": [0.724, 0.141, 0.065, 0.022, 0.815, None],
        "frontier": [0.870, 0.109, 0.098, 0.054, 0.848, None],
    },
}

METRICS = [
    ("recall", "up", "of names that should be tagged, how many were caught (safety-critical)"),
    ("leakage", "down", "share of real names missed — the direct privacy risk"),
    ("over_tag", "down", "how often a non-name was tagged"),
    ("integrity", "down", "output-minus-tags ≠ input (any other change to the text)"),
    ("pass", "up", "fully-correct passages (all-and-only names tagged, byte-identical otherwise)"),
    ("consistency", "up", "same answer across rewordings of the same case"),
]

FOOTNOTES = {
    "†": "The authored heldout_names row is a bf16 MPS carry-over (sft-v3-mps), not a genuine "
    "4-bit Colab result — the 4-bit eval on this set was never run. Read it as a bf16 "
    "reference, not an apples-to-apples 4-bit number. See final-report.md §3.3.",
}


def load_set(path: str) -> list[dict]:
    rows = []
    for line in (REPO / path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_payload() -> dict:
    sets = []
    for key, path, blurb, source in SETS:
        rows = load_set(path)
        # keep only the fields the viewer needs, verbatim
        cases = [
            {
                "id": r.get("id"),
                "input": r.get("input"),
                "target": r.get("target"),
                "category": r.get("category"),
                "register": r.get("register"),
                "ambiguous_token": r.get("ambiguous_token"),
                "n_spans": sum(1 for s in r.get("spans", []) if s.get("is_name")),
            }
            for r in rows
        ]
        sets.append(
            {
                "key": key,
                "path": path,
                "blurb": blurb,
                "source": source,
                "n": len(cases),
                "cases": cases,
            }
        )
    return {
        "sets": sets,
        "engines": [{"key": k, "desc": d} for k, d in ENGINES],
        "metrics": [{"key": k, "dir": d, "desc": t} for k, d, t in METRICS],
        "results": RESULTS,
        "footnotes": FOOTNOTES,
        "prompt": {"system": SYSTEM_PROMPT},
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>De-Id SLM — Eval Suite</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --panel2: #1e222b; --line: #2a2f3a;
    --ink: #e8eaed; --muted: #9aa3b2; --faint: #6b7280;
    --accent: #7aa2f7; --good: #4ade80; --bad: #f87171; --warn: #fbbf24;
    --tag-bg: rgba(122,162,247,.18); --tag-line: #7aa2f7;
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  a { color: var(--accent); }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 28px 22px 80px; }
  header h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -.01em; }
  header p.sub { color: var(--muted); margin: 0 0 22px; max-width: 780px; }
  .kpis { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 26px; }
  .kpi { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
         padding: 10px 14px; min-width: 92px; }
  .kpi b { display: block; font-size: 20px; }
  .kpi span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
       border-bottom: 1px solid var(--line); padding-bottom: 8px; margin: 34px 0 16px; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  thead th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase;
             letter-spacing: .03em; cursor: default; }
  tbody tr:hover { background: var(--panel2); }
  .num { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }
  .cell { border-radius: 5px; padding: 2px 6px; }
  .eng { font-weight: 600; }
  .eng small { display: block; font-weight: 400; color: var(--faint); font-size: 11px; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px;
          border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
  /* set tabs */
  .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
  .tab { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
         padding: 9px 14px; cursor: pointer; color: var(--ink); text-align: left; }
  .tab.active { border-color: var(--accent); background: var(--panel2); }
  .tab b { font-size: 14px; }
  .tab span { display: block; color: var(--faint); font-size: 12px; }
  .setmeta { color: var(--muted); margin: -4px 0 18px; max-width: 820px; }
  /* controls */
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 14px 0; }
  input[type=search] { background: var(--panel); border: 1px solid var(--line); color: var(--ink);
    border-radius: 8px; padding: 8px 12px; min-width: 240px; font-size: 14px; }
  .chip { background: var(--panel); border: 1px solid var(--line); color: var(--muted);
    border-radius: 999px; padding: 4px 11px; cursor: pointer; font-size: 12px; }
  .chip.active { border-color: var(--accent); color: var(--ink); background: var(--panel2); }
  /* case cards */
  .cases { display: grid; gap: 10px; }
  .case { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 13px 15px; }
  .case .top { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
  .case .id { color: var(--faint); font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .passage { font-size: 15px; line-height: 1.7; }
  .passage.raw { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; color: var(--muted); }
  .nm { background: var(--tag-bg); border-bottom: 2px solid var(--tag-line); border-radius: 3px 3px 0 0;
        padding: 0 2px; font-weight: 600; }
  .cat { font-size: 11px; }
  .cat-person_vs_eponym { color: #f0abfc; } .cat-person_vs_place { color: #7dd3fc; }
  .cat-person_vs_common { color: #fcd34d; } .cat-negative_trap { color: #fca5a5; }
  .cat-first_name_only { color: #86efac; } .cat-possessive { color: #c4b5fd; }
  .cat-third_party { color: #f9a8d4; } .cat-easy { color: #9ca3af; }
  .cat-adversarial { color: #fdba74; }
  .empty { color: var(--faint); padding: 20px 0; }
  .foot { color: var(--faint); font-size: 12px; margin-top: 10px; }
  .toggle { margin-left: auto; }
  .legend { color: var(--faint); font-size: 12px; margin: 4px 0 0; }
  /* base-vs-tuned real-output comparison */
  .cmp-agg { border-collapse: collapse; width: 100%; max-width: 640px; font-size: 14px; margin: 4px 0 8px; }
  .cmp-agg th, .cmp-agg td { padding: 7px 12px; border-bottom: 1px solid var(--line); text-align: right; }
  .cmp-agg th:first-child, .cmp-agg td:first-child { text-align: left; color: var(--muted); }
  .cmp-agg thead th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
  .cmp-agg .b { color: var(--bad); font-variant-numeric: tabular-nums; }
  .cmp-agg .t { color: var(--good); font-weight: 600; font-variant-numeric: tabular-nums; }
  .cmp-agg td small { color: var(--faint); font-weight: 400; }
  .cmp { display: grid; gap: 12px; }
  .cmp-card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
  .cmp-card.failtune { border-color: var(--warn); }
  .cmp-head { padding: 11px 15px; border-bottom: 1px solid var(--line); display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
  .cmp-head .intended { color: var(--muted); font-size: 13px; }
  .cmp-head .intended b { color: var(--ink); }
  .cmp-grid { display: grid; grid-template-columns: 1fr 1fr; }
  @media (max-width: 680px) { .cmp-grid { grid-template-columns: 1fr; } }
  .cmp-col { padding: 13px 15px; }
  .cmp-col.base { border-right: 1px solid var(--line); }
  @media (max-width: 680px) { .cmp-col.base { border-right: none; border-bottom: 1px solid var(--line); } }
  .cmp-lab { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
             margin-bottom: 8px; display: flex; gap: 8px; align-items: center; }
  .cmp-out { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px;
             line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
  .mk { background: var(--tag-bg); color: var(--tag-line); border-radius: 4px; padding: 0 3px; font-weight: 600; }
  .badge { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; }
  .badge.pass { color: var(--good); background: rgba(74,222,128,.14); }
  .badge.fail { color: var(--bad); background: rgba(248,113,113,.14); }
  .flags { margin-top: 9px; display: flex; gap: 6px; flex-wrap: wrap; }
  .flag { font-size: 11px; padding: 1px 8px; border-radius: 999px; background: rgba(248,113,113,.14); color: var(--bad); }
  .flag.ok { background: rgba(74,222,128,.14); color: var(--good); }
  .cmp-why { font-size: 12.5px; color: var(--muted); padding: 10px 15px; border-top: 1px solid var(--line); background: var(--panel2); }
  /* prompt panel */
  .prompt { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); margin: 4px 0 16px; }
  .prompt > summary { cursor: pointer; padding: 11px 15px; color: var(--ink); font-size: 13px; font-weight: 600; list-style: none; }
  .prompt > summary::-webkit-details-marker { display: none; }
  .prompt > summary::before { content: "\25B8"; color: var(--faint); margin-right: 8px; display: inline-block; transition: transform .15s; }
  .prompt[open] > summary::before { transform: rotate(90deg); }
  .prompt .pbody { padding: 0 15px 14px; }
  .prompt .role { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--accent); margin: 6px 0 4px; }
  .promptpre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; line-height: 1.6;
               white-space: pre-wrap; word-break: break-word; background: var(--bg); border: 1px solid var(--line);
               border-radius: 8px; padding: 11px 13px; margin: 0 0 6px; color: var(--ink); }
  .promptpre .nm { background: var(--tag-bg); color: var(--tag-line); border: none; border-radius: 4px; padding: 0 3px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>De-Id SLM &mdash; Eval Suite</h1>
    <p class="sub">The judgment core of de-identification: decide, in context, whether a token is a
      <b>person&rsquo;s name</b> or an identically-spelled non-person, and tag only the names
      (<span class="nm">&#10216;NAME&#10217;&hellip;&#10216;/NAME&#10217;</span>) leaving the passage byte-identical.
      Below: the five quarantined test sets (the cases) and how four engines score on each (the results).</p>
  </header>

  <div class="kpis" id="kpis"></div>

  <h2>Results &mdash; pass rate, 4 engines &times; 5 sets</h2>
  <p class="legend">Fully-correct passages. Fine-tunes (authored, gpt551) vs. the prompted base and the
    frontier ceiling (gpt-4.1). Greener = higher. Source: <code>docs/final-report.md</code> &sect;3.1.</p>
  <table id="matrix"></table>

  <h2>Base vs fine-tuned &mdash; real model outputs</h2>
  <p class="legend">What the numbers look like on the page. Every string below is a <b>real generated
    output</b> copied verbatim from the eval report pair <code id="cmp-src"></code> &mdash; not gold
    targets, not hand-edited. The tune tags the right span and changes nothing else; the base leaks,
    over-tags, and breaks byte-integrity. One honest tuned failure is kept in (per <code>CLAUDE.md</code>).</p>
  <details class="prompt" id="promptbox" open>
    <summary>The prompt &mdash; identical for both models (only the weights differ)</summary>
    <div class="pbody">
      <p class="legend" style="margin:0 0 10px">Both the prompted base and the fine-tune are called the exact same
        way (<code>src/infer.py</code> &rarr; <code>build_messages</code>, the same form training used): a fixed
        <b>system</b> message + a <b>user</b> turn that is the raw passage, byte-for-byte. Nothing else changes between
        the two runs &mdash; so every difference above is the fine-tune, not the prompt.</p>
      <div class="role">system</div>
      <pre class="promptpre" id="promptsys"></pre>
      <div class="role">user</div>
      <pre class="promptpre">&laquo; the passage, verbatim &raquo;   e.g.  Newton stayed late in the lab because the experiment kept failing.</pre>
    </div>
  </details>
  <table class="cmp-agg" id="cmp-agg"></table>
  <div class="cmp" id="cmp"></div>

  <h2>Browse a set</h2>
  <div class="tabs" id="tabs"></div>
  <p class="setmeta" id="setmeta"></p>

  <h3 style="color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.06em;">Scores on this set</h3>
  <p class="legend">Green = good, red = bad, shaded by direction (recall/pass/consistency higher is better;
    leakage/over&#95;tag/integrity lower is better).</p>
  <table id="setmetrics"></table>
  <div class="foot" id="footnotes"></div>

  <h3 style="color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.06em;">Cases</h3>
  <div class="controls">
    <input type="search" id="search" placeholder="search passages, ids, tokens…" autocomplete="off">
    <div id="catchips"></div>
    <label class="toggle"><input type="checkbox" id="rawtoggle"> show raw target</label>
  </div>
  <div class="cases" id="cases"></div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// render a target string, turning tags into highlighted spans
function renderTagged(t) {
  if (t == null) return '';
  return esc(t)
    .replaceAll('⟨NAME⟩', '<span class="nm">')
    .replaceAll('⟨/NAME⟩', '</span>');
}
function fmt(v){ return v==null ? '–' : v.toFixed(3); }
function metricVal(cell){ return Array.isArray(cell) ? cell[0] : cell; }
function metricMark(cell){ return Array.isArray(cell) ? cell[1] : ''; }

// green->red scale by goodness in [0,1] (1 = best)
function heat(goodness){
  if (goodness==null) return 'transparent';
  const g = Math.max(0, Math.min(1, goodness));
  // interpolate red(248,113,113) -> panel -> green(74,222,128) with low alpha
  if (g >= .5){ const a = (g-.5)*2*0.32; return `rgba(74,222,128,${a.toFixed(3)})`; }
  const a = (.5-g)*2*0.32; return `rgba(248,113,113,${a.toFixed(3)})`;
}

const KPI = document.getElementById('kpis');
const totalCases = DATA.sets.reduce((a,s)=>a+s.n,0);
[['5','test sets'],[String(totalCases),'cases'],['4','engines'],['6','metrics'],['0','eval leak']]
  .forEach(([b,s])=>{ const d=document.createElement('div'); d.className='kpi';
    d.innerHTML=`<b class="num">${b}</b><span>${s}</span>`; KPI.appendChild(d); });

// --- headline pass matrix ---
(function(){
  const t = document.getElementById('matrix');
  const passIdx = DATA.metrics.findIndex(m=>m.key==='pass');
  let head = '<thead><tr><th>set (n)</th>';
  DATA.engines.forEach(e=> head += `<th title="${esc(e.desc)}">${e.key}</th>`);
  head += '</tr></thead>';
  let body = '<tbody>';
  DATA.sets.forEach(s=>{
    body += `<tr><td>${s.key} <span class="pill">${s.n}</span></td>`;
    DATA.engines.forEach(e=>{
      const cell = DATA.results[s.key][e.key][passIdx];
      const v = metricVal(cell), mk = metricMark(cell);
      body += `<td class="num"><span class="cell" style="background:${heat(v)}">${fmt(v)}${mk?'<sup>'+mk+'</sup>':''}</span></td>`;
    });
    body += '</tr>';
  });
  body += '</tbody>';
  t.innerHTML = head + body;
})();

// --- set tabs ---
let activeSet = DATA.sets[0].key;
let activeCat = null;
const tabsEl = document.getElementById('tabs');
DATA.sets.forEach(s=>{
  const b = document.createElement('button');
  b.className = 'tab' + (s.key===activeSet?' active':'');
  b.innerHTML = `<b>${s.key}</b><span>${s.n} cases &middot; ${esc(s.source)}</span>`;
  b.onclick = ()=>{ activeSet=s.key; activeCat=null; document.getElementById('search').value=''; render(); };
  tabsEl.appendChild(b);
});

function currentSet(){ return DATA.sets.find(s=>s.key===activeSet); }

function renderSetMeta(){
  const s = currentSet();
  document.getElementById('setmeta').innerHTML =
    `<b>${s.key}</b> &mdash; ${esc(s.blurb)} <span class="pill">source: ${esc(s.source)}</span>`;
}

function renderSetMetrics(){
  const s = currentSet();
  const t = document.getElementById('setmetrics');
  let head = '<thead><tr><th>engine</th>';
  DATA.metrics.forEach(m=> head += `<th title="${esc(m.desc)} (${m.dir==='up'?'higher':'lower'} is better)">${m.key}</th>`);
  head += '</tr></thead><tbody>';
  const marks = new Set();
  DATA.engines.forEach(e=>{
    head += `<tr><td class="eng">${e.key}<small>${esc(e.desc)}</small></td>`;
    DATA.metrics.forEach((m,i)=>{
      const cell = DATA.results[s.key][e.key][i];
      const v = metricVal(cell), mk = metricMark(cell);
      if (mk) marks.add(mk);
      const goodness = v==null ? null : (m.dir==='up' ? v : 1-v);
      head += `<td class="num"><span class="cell" style="background:${heat(goodness)}">${fmt(v)}${mk?'<sup>'+mk+'</sup>':''}</span></td>`;
    });
    head += '</tr>';
  });
  head += '</tbody>';
  t.innerHTML = head;
  const fn = document.getElementById('footnotes');
  fn.innerHTML = [...marks].map(m=> `<div>${m} ${esc(DATA.footnotes[m]||'')}</div>`).join('');
}

function renderCatChips(){
  const s = currentSet();
  const counts = {};
  s.cases.forEach(c=> counts[c.category]=(counts[c.category]||0)+1);
  const el = document.getElementById('catchips');
  el.innerHTML = '';
  const mk = (label, cat, n)=>{ const c=document.createElement('button');
    c.className='chip'+((cat===activeCat)||(cat===null&&activeCat===null)?' active':'');
    c.innerHTML = `${label} <b class="num">${n}</b>`;
    c.onclick=()=>{ activeCat=cat; render(); }; el.appendChild(c); };
  mk('all', null, s.cases.length);
  Object.keys(counts).sort().forEach(cat=> mk(cat, cat, counts[cat]));
}

function renderCases(){
  const s = currentSet();
  const q = document.getElementById('search').value.trim().toLowerCase();
  const raw = document.getElementById('rawtoggle').checked;
  const list = s.cases.filter(c=>{
    if (activeCat && c.category!==activeCat) return false;
    if (!q) return true;
    return (c.input||'').toLowerCase().includes(q)
        || (c.id||'').toLowerCase().includes(q)
        || (c.ambiguous_token||'').toLowerCase().includes(q);
  });
  const el = document.getElementById('cases');
  if (!list.length){ el.innerHTML = '<div class="empty">No cases match.</div>'; return; }
  el.innerHTML = list.map(c=>{
    const body = raw ? `<div class="passage raw">${esc(c.target)}</div>`
                     : `<div class="passage">${renderTagged(c.target)}</div>`;
    const tok = c.ambiguous_token ? `<span class="pill">token: ${esc(c.ambiguous_token)}</span>` : '';
    const nspan = c.n_spans>0 ? `<span class="pill">${c.n_spans} name${c.n_spans>1?'s':''}</span>`
                              : `<span class="pill">no names</span>`;
    return `<div class="case">
      <div class="top">
        <span class="cat cat-${esc(c.category)}">&#9679; ${esc(c.category)}</span>
        <span class="pill">${esc(c.register)}</span>
        ${nspan} ${tok}
        <span class="id" style="margin-left:auto">${esc(c.id)}</span>
      </div>${body}</div>`;
  }).join('');
}

function render(){
  document.querySelectorAll('.tab').forEach((b,i)=>
    b.classList.toggle('active', DATA.sets[i].key===activeSet));
  renderSetMeta(); renderSetMetrics(); renderCatChips(); renderCases();
}
document.getElementById('search').addEventListener('input', renderCases);
document.getElementById('rawtoggle').addEventListener('change', renderCases);

// --- base-vs-tuned real outputs (from outputs/eval_reports_colab_run2) ---
const COMPARE = {
  src: 'outputs/eval_reports_colab_run2 · base vs tuned-v3 (4-bit QLoRA, Colab T4)',
  agg: [
    ['pass rate (fully correct)',            0.35, 0.88, 'up'],
    ['F5 (recall-weighted)',                 0.51, 0.92, 'up'],
    ['leakage rate',    0.25, 0.04, 'down', 'name left untagged'],
    ['over-tag rate',   0.55, 0.10, 'down', 'non-name tagged'],
    ['integrity violations', 0.59, 0.00, 'down', 'text altered beyond tags'],
    ['consistency across paraphrases',       0.38, 0.88, 'up'],
  ],
  cases: [
    { cat:'person_vs_eponym', intended:'<b>tag</b> — "Newton" is the person here',
      base:{ out:'⟨NAME⟩Newton ⟨NAME⟩ stayed late in the lab because the experiment kept failing.', pass:false, flags:['leaked','integrity broken'] },
      tuned:{ out:'⟨NAME⟩Newton⟨/NAME⟩ stayed late in the lab because the experiment kept failing.', pass:true, flags:[] },
      why:'Base emits a malformed open/open tag pair — the name is effectively leaked and the byte stream is corrupted. The tune produces a well-formed span.' },
    { cat:'person_vs_eponym', intended:'<b>don\'t tag</b> — "Newton method" is a technique, not a person',
      base:{ out:'⟨NAME⟩We⟨/NAME⟩ applied the Newton method to approximate the root of the equation.', pass:false, flags:['over-tagged'] },
      tuned:{ out:'We applied the Newton method to approximate the root of the equation.', pass:true, flags:[] },
      why:'Base tags the wrong token entirely ("We") and leaves the eponym. The tune correctly tags nothing.' },
    { cat:'possessive', intended:'<b>don\'t tag</b> — "Newton\'s laws" is an eponymous law, not the person',
      base:{ out:'⟨NAME⟩...⟨/NAME⟩', pass:false, flags:['over-tagged','integrity broken'] },
      tuned:{ out:'Newton\'s laws describe the motion of classical objects.', pass:true, flags:[] },
      why:'The possessive of an eponym is still not a person mention. Base collapses to an empty placeholder; the tune leaves the sentence alone.' },
    { cat:'person_vs_eponym', intended:'<b>don\'t tag</b> — "gauss" is a unit of measurement',
      base:{ out:'⟨NAME⟩...⟨/NAME⟩', pass:false, flags:['over-tagged','integrity broken'] },
      tuned:{ out:'The magnetic field strength was measured in gauss during the experiment.', pass:true, flags:[] },
      why:'A unit named after a scientist is not a person. Base emits a garbage placeholder that destroys the text; the tune keeps it byte-identical.' },
    { cat:'person_vs_place', intended:'<b>tag</b> — "Chelsea" is a person being thanked',
      base:{ out:'⟨NAME⟩Chelsea⟨/NAME⟩', pass:false, flags:['leaked','over-tagged','integrity broken'] },
      tuned:{ out:'thanks to ⟨NAME⟩Chelsea⟨/NAME⟩ I finally understood the proof', pass:true, flags:[] },
      why:'Base collapses the whole passage down to just the tagged word, destroying the surrounding text. The tune tags in place and preserves every other byte.' },
    { cat:'person_vs_place', intended:'<b>don\'t tag</b> — "Chelsea" is a neighbourhood here',
      base:{ out:'⟨NAME⟩Chelsea⟨/NAME⟩', pass:false, flags:['over-tagged','integrity broken'] },
      tuned:{ out:'Last summer I visited Chelsea and walked along the river all afternoon.', pass:true, flags:[] },
      why:'Same surface word, opposite correct call. Base tags it (and again eats the passage); the tune reads it as a place and leaves it alone.' },
    { cat:'person_vs_place', intended:'<b>don\'t tag</b> — "Florence" is the city',
      base:{ out:'⟨NAME⟩...⟨/NAME⟩', pass:false, flags:['over-tagged','integrity broken'] },
      tuned:{ out:'The Renaissance art in Florence left a deep impression on the class.', pass:true, flags:[] },
      why:'"Renaissance art in Florence" is unmistakably the city. Base wrecks the passage; the tune reads the context and leaves it.' },
    { cat:'person_vs_common', intended:'<b>tag</b> — "Bishop" is the TA’s name',
      base:{ out:'⟨NAME⟩Bishop⟨/NAME⟩', pass:false, flags:['leaked','over-tagged','integrity broken'] },
      tuned:{ out:'our TA ⟨NAME⟩Bishop⟨/NAME⟩ regraded my problem set', pass:true, flags:[] },
      why:'"our TA Bishop" makes Bishop a person. Base tags the right word but eats the rest of the sentence; the tune tags in place.' },
    { cat:'person_vs_common', intended:'<b>don\'t tag</b> — "bishop" is the chess piece (same word as the card above)',
      base:{ out:'⟨NAME⟩Bishop⟨/NAME⟩', pass:false, flags:['over-tagged','integrity broken'] },
      tuned:{ out:'I moved my bishop to threaten the queen in the endgame.', pass:true, flags:[] },
      why:'The mirror image of the previous card: identical surface word, opposite call. "moved my bishop … endgame" is chess — the tune tags nothing.' },
    { cat:'person_vs_common', intended:'<b>tag</b> — "Baker" is a professor’s surname',
      base:{ out:'⟨NAME⟩Professor Baker⟨/NAME⟩ extended the deadline for the whole class.', pass:false, flags:['leaked','over-tagged'] },
      tuned:{ out:'Professor ⟨NAME⟩Baker⟨/NAME⟩ extended the deadline for the whole class.', pass:true, flags:[] },
      why:'Base pulls the title into the span ("Professor Baker") — an over-tag that also misses the exact gold name. The tune tags just "Baker".' },
    { cat:'person_vs_common', intended:'<b>don\'t tag</b> — "baker" is an occupation here (same word as above)',
      base:{ out:'⟨NAME⟩...⟨/NAME⟩', pass:false, flags:['over-tagged','integrity broken'] },
      tuned:{ out:'The baker down the street sponsored our fundraising bake sale.', pass:true, flags:[] },
      why:'Lower-case "the baker down the street" is a job, not a name. The tune leaves it; base emits a placeholder.' },
    { cat:'person_vs_common', intended:'<b>don\'t tag</b> — "rose" is a flower',
      base:{ out:'⟨NAME⟩...⟨/NAME⟩', pass:false, flags:['over-tagged','integrity broken'] },
      tuned:{ out:'The rose in the courtyard bloomed early this year.', pass:true, flags:[] },
      why:'Base degenerates into an empty tagged placeholder. The tune recognises "rose" as a common noun and returns the passage untouched.' },
    { cat:'first_name_only', intended:'<b>tag</b> — "Sam" is a first-name-only address',
      base:{ out:'⟨NAME⟩Sam⟨/NAME⟩ — that explanation finally clicked', pass:false, flags:['leaked','over-tagged','integrity broken'] },
      tuned:{ out:'thanks, ⟨NAME⟩Sam⟨/NAME⟩ — that explanation finally clicked', pass:true, flags:[] },
      why:'Base tags "Sam" correctly but drops the leading "thanks," — an integrity violation. The tune tags the name and keeps the sentence intact.' },
    { cat:'first_name_only', intended:'<b>tag</b> — "Alvarez" is a name buried mid-sentence',
      base:{ out:'⟨NAME⟩...⟨/NAME⟩', pass:false, flags:['leaked','over-tagged','integrity broken'] },
      tuned:{ out:'When ⟨NAME⟩Alvarez⟨/NAME⟩ presented, the room finally went quiet.', pass:true, flags:[] },
      why:'Base garbles into a placeholder, leaking the name and wrecking the text. The tune finds the name after "When" and tags only it.' },
    { cat:'third_party', intended:'<b>tag</b> — "Priya" is a named third person',
      base:{ out:'⟨NAME⟩...⟨/NAME⟩', pass:false, flags:['leaked','over-tagged','integrity broken'] },
      tuned:{ out:'My mom, ⟨NAME⟩Priya⟨/NAME⟩, quizzed me on vocabulary every evening.', pass:true, flags:[] },
      why:'A parent named in passing. Base produces a placeholder; the tune tags "Priya" and preserves the commas and the rest.' },
    { cat:'easy', intended:'<b>tag</b> — a full name in a sign-off',
      base:{ out:'⟨NAME⟩Maria Gonzalez⟨/NAME⟩', pass:false, flags:['leaked','over-tagged','integrity broken'] },
      tuned:{ out:'Sincerely, ⟨NAME⟩Maria Gonzalez⟨/NAME⟩', pass:true, flags:[] },
      why:'Even an easy full-name sign-off trips the base: it tags the name but drops "Sincerely," (integrity break). The tune keeps the whole line.' },
    { cat:'negative_trap', intended:'<b>don\'t tag</b> — "Android" is a brand, not a person',
      base:{ out:'<i>wrote</i> the whole draft on my Android phone during the bus ride', pass:false, flags:['integrity broken'] },
      tuned:{ out:'i wrote the whole draft on my Android phone during the bus ride', pass:true, flags:[] },
      why:'Base hallucinates stray HTML markup and mangles the opening — a hard integrity failure. The tune returns the passage byte-for-byte.' },
    { cat:'negative_trap', intended:'<b>don\'t tag</b> — "March"/"April" are months. <b>Honest failure: the tune gets this wrong.</b>', failtune:true,
      base:{ out:'March felt endless, but April brought the final presentations.', pass:true, flags:[] },
      tuned:{ out:'⟨NAME⟩March⟨/NAME⟩ felt endless, but ⟨NAME⟩April⟨/NAME⟩ brought the final presentations.', pass:false, flags:['over-tagged'] },
      why:'The one shown case where the base wins: months that double as given names fool the tune into over-tagging — its main residual error mode. Left in deliberately.' },
  ],
};

// render marker tokens as visible chips (robust to malformed / unbalanced tags)
function renderMarks(t){
  return esc(t)
    .replaceAll('⟨NAME⟩', '<span class="mk">⟨NAME⟩</span>')
    .replaceAll('⟨/NAME⟩', '<span class="mk">⟨/NAME⟩</span>');
}
(function(){
  document.getElementById('promptsys').innerHTML = renderTagged(DATA.prompt.system);
  document.getElementById('cmp-src').textContent = COMPARE.src;
  const at = document.getElementById('cmp-agg');
  let h = '<thead><tr><th>metric &mdash; 51 hard cases</th><th>base</th><th>fine-tuned</th></tr></thead><tbody>';
  COMPARE.agg.forEach(([label, b, tv, dir, hint])=>{
    const arrow = dir==='down' ? ' <small>&darr; lower better</small>' : '';
    h += `<tr><td>${esc(label)}${hint?` <small>(${esc(hint)})</small>`:''}${arrow}</td>`
       + `<td class="b">${b.toFixed(2)}</td><td class="t">${tv.toFixed(2)}</td></tr>`;
  });
  h += '</tbody>';
  at.innerHTML = h;

  const col = (label, side)=>{
    const badge = side.pass ? '<span class="badge pass">PASS</span>' : '<span class="badge fail">FAIL</span>';
    const flags = side.flags.length
      ? side.flags.map(f=>`<span class="flag">${esc(f)}</span>`).join('')
      : '<span class="flag ok">clean</span>';
    return `<div class="cmp-lab">${label} ${badge}</div>`
         + `<div class="cmp-out">${renderMarks(side.out)}</div>`
         + `<div class="flags">${flags}</div>`;
  };
  document.getElementById('cmp').innerHTML = COMPARE.cases.map(c=>
    `<div class="cmp-card${c.failtune?' failtune':''}">
      <div class="cmp-head">
        <span class="cat cat-${esc(c.cat)}">&#9679; ${esc(c.cat)}</span>
        <span class="intended">Intended: ${c.intended}</span>
      </div>
      <div class="cmp-grid">
        <div class="cmp-col base">${col('base', c.base)}</div>
        <div class="cmp-col">${col('fine-tuned', c.tuned)}</div>
      </div>
      <div class="cmp-why">${esc(c.why)}</div>
    </div>`).join('');
})();

render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "docs" / "eval-suite.html"))
    args = ap.parse_args()

    payload = build_payload()
    # embed as JSON; escape </script to keep the inline script intact
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", data_json)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    total = sum(s["n"] for s in payload["sets"])
    print(
        f"wrote {out}  ({total} cases across {len(payload['sets'])} sets, {out.stat().st_size:,} bytes)"
    )
    for s in payload["sets"]:
        print(f"  {s['key']:<15} {s['n']:>3} cases")


if __name__ == "__main__":
    main()
