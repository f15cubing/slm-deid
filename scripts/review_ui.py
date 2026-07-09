#!/usr/bin/env python3
"""Local web UI to review — and human-approve — the de-id dataset splits.

Browse any ``*.jsonl`` split (train / val / hardcases / cooccur), see the ``target``
rendered with ⟨NAME⟩ tags highlighted, inspect gold spans + metadata, filter by
category/register/source, full-text search, and — the point of "make sure it's good" —
every record is run through :meth:`schema.Example.validate` so schema/integrity
violations are flagged right in the list.

Human-in-the-loop review:
  - Approve / Deny each item (optional note); decisions persist server-side under
    ``reviews/<split>.approvals.json`` so they survive restarts.
  - Keyboard: J/K move, A approve, D deny, U clear, N focus note.
  - Filter by decision (approved / denied / pending) and watch live progress.
  - "Seal approved" writes an approved-only JSONL next to the reviews file.

    python scripts/review_ui.py            # serve http://127.0.0.1:8765
    python scripts/review_ui.py --port 9000
    python scripts/review_ui.py --no-open  # don't auto-open a browser

No third-party deps — stdlib ``http.server`` only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Import the project's schema so validation matches the real pipeline exactly.
from src.common import schema, tags  # noqa: E402

# Directories we scan for reviewable splits, relative to the repo root.
SCAN_GLOBS = [
    "data/splits/*.jsonl",
    "eval/**/*.jsonl",
    "data/raw/*.jsonl",
    "data/cooccur/*.jsonl",
]

# Where human approve/deny decisions are persisted (one JSON file per split).
REVIEWS_DIR = REPO / "reviews"
_review_lock = threading.Lock()
DECISIONS = {"approved", "denied", ""}


def discover_files() -> list[dict]:
    """Return metadata for every reviewable JSONL under the repo (sorted, de-duped)."""
    seen: set[Path] = set()
    out: list[dict] = []
    for pattern in SCAN_GLOBS:
        for p in sorted(REPO.glob(pattern)):
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            rel = p.relative_to(REPO).as_posix()
            try:
                n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
            except OSError:
                n = -1
            out.append({"path": rel, "count": n})
    return out


# --- review persistence --------------------------------------------------------------
def _safe_rel(rel_path: str) -> Path:
    """Resolve a repo-relative path, refusing anything that escapes the repo."""
    target = (REPO / rel_path).resolve()
    if REPO not in target.parents:
        raise ValueError("path escapes repo")
    return target


def _reviews_file(rel_path: str) -> Path:
    slug = rel_path.replace("/", "__").replace("\\", "__")
    return REVIEWS_DIR / (slug + ".approvals.json")


def load_reviews(rel_path: str) -> dict:
    """Return the persisted ``{record_key: {decision, note, updated}}`` map (may be empty)."""
    p = _reviews_file(rel_path)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_review(rel_path: str, key: str, decision: str, note: str) -> dict | None:
    """Persist one decision. Empty decision + empty note clears the record. Thread-safe."""
    if decision not in DECISIONS:
        raise ValueError(f"bad decision {decision!r}")
    with _review_lock:
        data = load_reviews(rel_path)
        if not decision and not note:
            data.pop(key, None)
            entry = None
        else:
            entry = {
                "decision": decision,
                "note": note or "",
                "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            data[key] = entry
        REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        _reviews_file(rel_path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return entry


def _record_key(raw: dict, line: int) -> str:
    rid = raw.get("id")
    return str(rid) if rid else f"#L{line}"


def load_records(rel_path: str) -> dict:
    """Parse one split, validate each row, merge in saved decisions, return records + stats."""
    target = _safe_rel(rel_path)
    if not target.is_file():
        raise FileNotFoundError(rel_path)

    reviews = load_reviews(rel_path)
    records: list[dict] = []
    stats = {
        "total": 0, "valid": 0, "invalid": 0, "with_spans": 0, "quarantine": 0,
        "approved": 0, "denied": 0, "pending": 0,
        "categories": {}, "registers": {}, "sources": {},
    }

    with target.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            rec: dict = {"line": i}
            try:
                ex = schema.loads(line)
                rec["raw"] = ex.to_dict()
                try:
                    ex.validate()
                    rec["valid"] = True
                    rec["error"] = None
                    stats["valid"] += 1
                except schema.SchemaError as e:
                    rec["valid"] = False
                    rec["error"] = str(e)
                    stats["invalid"] += 1
                rec["segments"] = render_segments(ex.target)
                d = ex.to_dict()
                if any(sp.get("is_name") for sp in d.get("spans", [])):
                    stats["with_spans"] += 1
                if d.get("quarantine"):
                    stats["quarantine"] += 1
                _bump(stats["categories"], d.get("category"))
                _bump(stats["registers"], d.get("register"))
                _bump(stats["sources"], d.get("source"))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                rec["valid"] = False
                rec["error"] = f"parse error: {e}"
                rec["raw"] = {"_unparsed": line[:500]}
                rec["segments"] = []
                stats["invalid"] += 1

            key = _record_key(rec["raw"], i)
            rec["key"] = key
            rec["review"] = reviews.get(key)
            decision = (rec["review"] or {}).get("decision", "")
            if decision == "approved":
                stats["approved"] += 1
            elif decision == "denied":
                stats["denied"] += 1
            else:
                stats["pending"] += 1
            records.append(rec)

    return {"path": rel_path, "records": records, "stats": stats}


def export_approved(rel_path: str) -> dict:
    """Write an approved-only JSONL (the 'seal'), skipping any row that fails validation."""
    target = _safe_rel(rel_path)
    if not target.is_file():
        raise FileNotFoundError(rel_path)
    reviews = load_reviews(rel_path)
    approved_keys = {k for k, v in reviews.items() if (v or {}).get("decision") == "approved"}

    kept: list[schema.Example] = []
    skipped_invalid = 0
    with target.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = schema.loads(line)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if _record_key(ex.to_dict(), i) not in approved_keys:
                continue
            try:
                ex.validate()
            except schema.SchemaError:
                skipped_invalid += 1
                continue
            kept.append(ex)

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    slug = rel_path.replace("/", "__").replace("\\", "__")
    out = REVIEWS_DIR / (slug + ".approved.jsonl")
    n = schema.write_jsonl(out, kept)
    return {
        "path": out.relative_to(REPO).as_posix(),
        "written": n,
        "approved_total": len(approved_keys),
        "skipped_invalid": skipped_invalid,
    }


def render_segments(target: str) -> list[dict]:
    """Split a tagged target into ``[{text, name}]`` segments for highlighting."""
    if not tags.is_well_formed(target):
        return [{"text": target, "name": False, "malformed": True}]
    segs: list[dict] = []
    pos = 0
    open_tag, close_tag = tags.NAME_OPEN, tags.NAME_CLOSE
    while True:
        o = target.find(open_tag, pos)
        if o == -1:
            if pos < len(target):
                segs.append({"text": target[pos:], "name": False})
            break
        if o > pos:
            segs.append({"text": target[pos:o], "name": False})
        c = target.find(close_tag, o + len(open_tag))
        if c == -1:
            segs.append({"text": target[o:], "name": False})
            break
        inner = target[o + len(open_tag) : c]
        segs.append({"text": inner, "name": True})
        pos = c + len(close_tag)
    return segs


def _bump(counter: dict, key) -> None:
    if key is None:
        key = "—"
    counter[key] = counter.get(key, 0) + 1


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(
            code,
            json.dumps(obj, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/files":
            self._json(discover_files())
            return
        if route == "/api/records":
            rel = (parse_qs(parsed.query).get("file") or [""])[0]
            if not rel:
                self._json({"error": "missing ?file="}, 400)
                return
            try:
                self._json(load_records(rel))
            except FileNotFoundError:
                self._json({"error": f"not found: {rel}"}, 404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        body = self._body()
        rel = body.get("file", "")
        if route == "/api/review":
            key = body.get("key", "")
            decision = body.get("decision", "")
            note = body.get("note", "")
            if not rel or not key:
                self._json({"error": "missing file/key"}, 400)
                return
            try:
                _safe_rel(rel)
                entry = save_review(rel, key, decision, note)
                self._json({"ok": True, "review": entry})
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            return
        if route == "/api/export":
            if not rel:
                self._json({"error": "missing file"}, 400)
                return
            try:
                self._json(export_approved(rel))
            except FileNotFoundError:
                self._json({"error": f"not found: {rel}"}, 404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            return
        self._json({"error": "not found"}, 404)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>De-id test-set review</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --border:#2a2f3a;
    --fg:#e6e9ef; --muted:#8b93a7; --accent:#6ea8fe; --accent2:#7ee787;
    --name-bg:#2b4d2b; --name-fg:#b9f6b9; --bad:#ff6b6b; --bad-bg:#3a1f22;
    --warn:#ffcc66; --ok:#3fb950; --deny:#f85149;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; }
  body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .app { display:grid; grid-template-columns:300px 1fr; height:100vh; }
  .sidebar { background:var(--panel); border-right:1px solid var(--border); padding:16px; overflow:auto; }
  .main { overflow:auto; padding:20px 24px; }
  h1 { font-size:15px; margin:0 0 4px; letter-spacing:.02em; }
  .hint { font-size:11px; color:var(--muted); margin:0 0 12px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:18px 0 8px; }
  select,input[type=text] { width:100%; background:var(--panel2); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:7px 9px; font:inherit; }
  label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  .chip { display:inline-block; padding:1px 7px; border-radius:999px; font-size:11px; background:var(--panel2); border:1px solid var(--border); color:var(--muted); margin:0 4px 4px 0; }
  .stat { display:flex; justify-content:space-between; padding:2px 0; font-size:12px; color:var(--muted); }
  .stat b { color:var(--fg); font-weight:600; }
  .stat.bad b { color:var(--bad); }
  .stat.ok b { color:var(--accent2); }
  .stat.appr b { color:var(--ok); }
  .stat.deny b { color:var(--deny); }
  .progress { height:8px; background:var(--panel2); border-radius:999px; overflow:hidden; margin:8px 0 4px; display:flex; }
  .progress .p-appr { background:var(--ok); }
  .progress .p-deny { background:var(--deny); }
  .card { background:var(--panel); border:1px solid var(--border); border-left-width:4px; border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  .card.invalid { border-color:var(--bad); background:var(--bad-bg); }
  .card.approved { border-left-color:var(--ok); }
  .card.denied { border-left-color:var(--deny); }
  .card.focused { outline:2px solid var(--accent); outline-offset:2px; }
  .card-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
  .id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--muted); }
  .badge { font-size:11px; padding:1px 8px; border-radius:6px; font-weight:600; }
  .badge.ok { background:#123; color:var(--accent2); border:1px solid #2c5; }
  .badge.err { background:#311; color:var(--bad); border:1px solid var(--bad); }
  .badge.q { background:#332; color:var(--warn); border:1px solid #664; }
  .badge.dec-approved { background:#0d2a14; color:var(--ok); border:1px solid var(--ok); }
  .badge.dec-denied { background:#2d0f0f; color:var(--deny); border:1px solid var(--deny); }
  .cat { background:var(--panel2); color:var(--accent); border:1px solid var(--border); }
  .text { white-space:pre-wrap; word-break:break-word; background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin:6px 0; }
  .text.rendered mark { background:var(--name-bg); color:var(--name-fg); padding:0 2px; border-radius:3px; font-weight:600; }
  .text.rendered mark::before { content:"⟨"; opacity:.5; }
  .text.rendered mark::after { content:"⟩"; opacity:.5; }
  .lbl { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-top:8px; }
  .err-msg { color:var(--bad); font-size:12px; margin-top:8px; font-family:ui-monospace,monospace; white-space:pre-wrap; }
  .spans { font-size:12px; color:var(--muted); margin-top:6px; }
  .spans code { background:var(--panel2); padding:1px 5px; border-radius:4px; color:var(--fg); }
  .spans .noname { opacity:.55; text-decoration:line-through; }
  .review-row { display:flex; gap:8px; align-items:center; margin-top:12px; flex-wrap:wrap; border-top:1px dashed var(--border); padding-top:12px; }
  .rbtn { border:1px solid var(--border); background:var(--panel2); color:var(--fg); border-radius:6px; padding:6px 14px; cursor:pointer; font:inherit; font-weight:600; }
  .rbtn.approve:hover, .rbtn.approve.active { background:var(--ok); border-color:var(--ok); color:#04220f; }
  .rbtn.deny:hover, .rbtn.deny.active { background:var(--deny); border-color:var(--deny); color:#2a0606; }
  .rbtn.clear:hover { border-color:var(--accent); }
  .note { flex:1; min-width:160px; }
  .updated { font-size:11px; color:var(--muted); }
  .toolbar { display:flex; gap:10px; align-items:center; margin-bottom:16px; flex-wrap:wrap; position:sticky; top:0; background:var(--bg); padding:6px 0; z-index:5; }
  .count { color:var(--muted); font-size:13px; }
  .empty { color:var(--muted); text-align:center; padding:60px 0; }
  .toggle { display:flex; gap:6px; }
  .btn { background:var(--panel2); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 12px; cursor:pointer; font:inherit; }
  .btn:hover { border-color:var(--accent); }
  .btn.seal { border-color:var(--ok); color:var(--ok); font-weight:600; }
  .toggle .btn.on { border-color:var(--accent); color:var(--accent); }
  .toast { position:fixed; bottom:18px; left:50%; transform:translateX(-50%); background:var(--panel2); border:1px solid var(--ok); color:var(--fg); padding:10px 16px; border-radius:8px; box-shadow:0 4px 20px rgba(0,0,0,.4); opacity:0; transition:opacity .2s; pointer-events:none; max-width:70vw; }
  .toast.show { opacity:1; }
  a { color:var(--accent); }
  kbd { background:var(--panel2); border:1px solid var(--border); border-bottom-width:2px; border-radius:4px; padding:0 5px; font-size:11px; font-family:ui-monospace,monospace; }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>De-id test-set review</h1>
    <p class="hint"><kbd>J</kbd>/<kbd>K</kbd> move · <kbd>A</kbd> approve · <kbd>D</kbd> deny · <kbd>U</kbd> clear · <kbd>N</kbd> note</p>
    <label for="file">Split</label>
    <select id="file"></select>

    <label for="search">Search text / id</label>
    <input type="text" id="search" placeholder="substring…" autocomplete="off">

    <label for="cat">Category</label>
    <select id="cat"><option value="">all</option></select>
    <label for="reg">Register</label>
    <select id="reg"><option value="">all</option></select>
    <label for="src">Source</label>
    <select id="src"><option value="">all</option></select>

    <label for="show">Show</label>
    <select id="show">
      <option value="all">all rows</option>
      <option value="pending">⬦ pending review</option>
      <option value="approved">✓ approved</option>
      <option value="denied">✗ denied</option>
      <option value="invalid">⚠ invalid only</option>
      <option value="spans">with name spans</option>
      <option value="nospans">no name spans (negatives)</option>
      <option value="quarantine">quarantined</option>
    </select>

    <h2>Review progress</h2>
    <div class="progress"><div class="p-appr" id="pAppr"></div><div class="p-deny" id="pDeny"></div></div>
    <div id="reviewStats"></div>

    <h2>Split stats</h2>
    <div id="stats"></div>
  </aside>

  <main class="main">
    <div class="toolbar">
      <span class="count" id="count"></span>
      <span style="flex:1"></span>
      <div class="toggle">
        <button class="btn on" id="viewRendered">Rendered</button>
        <button class="btn" id="viewDiff">Input vs target</button>
      </div>
      <button class="btn seal" id="seal" title="Write an approved-only .jsonl">🔏 Seal approved</button>
    </div>
    <div id="list"></div>
  </main>
</div>
<div class="toast" id="toast"></div>

<script>
const $ = s => document.querySelector(s);
let DATA = null;            // {records, stats}
let VIEW = "rendered";
let FILE = "";
let FILTERED = [];          // records currently shown (order = display)
let FOCUS = 0;              // index into FILTERED

async function j(url, opts){ const r = await fetch(url, opts); return r.json(); }
function post(url, body){ return j(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)}); }
function esc(s){ return (s??"").replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function toast(msg){ const t=$("#toast"); t.textContent=msg; t.classList.add("show"); clearTimeout(window._tt); window._tt=setTimeout(()=>t.classList.remove("show"), 3500); }

async function loadFiles(){
  const files = await j("/api/files");
  const sel = $("#file");
  sel.innerHTML = files.map(f=>`<option value="${esc(f.path)}">${esc(f.path)} (${f.count})</option>`).join("");
  const pref = files.find(f=>/val|test|cooccur/.test(f.path)) || files[0];
  if(pref){ sel.value = pref.path; await loadFile(pref.path); }
}

async function loadFile(path){
  FILE = path;
  DATA = await j("/api/records?file="+encodeURIComponent(path));
  if(DATA.error){ $("#list").innerHTML = `<div class="empty">${esc(DATA.error)}</div>`; return; }
  fillFilter("#cat", DATA.stats.categories);
  fillFilter("#reg", DATA.stats.registers);
  fillFilter("#src", DATA.stats.sources);
  renderStats();
  FOCUS = 0;
  render();
}

function fillFilter(sel, counts){
  const cur = $(sel).value;
  const opts = ['<option value="">all</option>'].concat(
    Object.entries(counts).sort((a,b)=>b[1]-a[1])
      .map(([k,v])=>`<option value="${esc(k)}">${esc(k)} (${v})</option>`));
  $(sel).innerHTML = opts.join("");
  if([...$(sel).options].some(o=>o.value===cur)) $(sel).value = cur;
}

function decisionOf(rec){ return (rec.review||{}).decision || ""; }

function recomputeReviewStats(){
  let a=0,d=0,p=0;
  for(const rec of DATA.records){
    const dec=decisionOf(rec);
    if(dec==="approved") a++; else if(dec==="denied") d++; else p++;
  }
  DATA.stats.approved=a; DATA.stats.denied=d; DATA.stats.pending=p;
}

function renderStats(){
  const s = DATA.stats;
  recomputeReviewStats();
  const tot=s.total||1;
  $("#pAppr").style.width = (100*s.approved/tot)+"%";
  $("#pDeny").style.width = (100*s.denied/tot)+"%";
  const reviewed = s.approved+s.denied;
  $("#reviewStats").innerHTML = `
    <div class="stat"><span>reviewed</span><b>${reviewed} / ${s.total}</b></div>
    <div class="stat appr"><span>✓ approved</span><b>${s.approved}</b></div>
    <div class="stat deny"><span>✗ denied</span><b>${s.denied}</b></div>
    <div class="stat"><span>⬦ pending</span><b>${s.pending}</b></div>`;
  $("#stats").innerHTML = `
    <div class="stat"><span>records</span><b>${s.total}</b></div>
    <div class="stat ok"><span>valid</span><b>${s.valid}</b></div>
    <div class="stat ${s.invalid?'bad':''}"><span>invalid</span><b>${s.invalid}</b></div>
    <div class="stat"><span>with name spans</span><b>${s.with_spans}</b></div>
    <div class="stat"><span>quarantined</span><b>${s.quarantine}</b></div>`;
}

function passFilters(rec){
  const raw = rec.raw||{};
  const cat=$("#cat").value, reg=$("#reg").value, src=$("#src").value, show=$("#show").value;
  const q=$("#search").value.trim().toLowerCase();
  if(cat && raw.category!==cat) return false;
  if(reg && raw.register!==reg) return false;
  if(src && raw.source!==src) return false;
  const hasName = (raw.spans||[]).some(sp=>sp.is_name);
  const dec = decisionOf(rec);
  if(show==="invalid" && rec.valid) return false;
  if(show==="spans" && !hasName) return false;
  if(show==="nospans" && hasName) return false;
  if(show==="quarantine" && !raw.quarantine) return false;
  if(show==="approved" && dec!=="approved") return false;
  if(show==="denied" && dec!=="denied") return false;
  if(show==="pending" && dec!=="") return false;
  if(q){
    const hay = ((raw.id||"")+" "+(raw.input||"")+" "+(raw.target||"")).toLowerCase();
    if(!hay.includes(q)) return false;
  }
  return true;
}

function segHTML(segments){
  return (segments||[]).map(s=> s.name ? `<mark>${esc(s.text)}</mark>` : esc(s.text)).join("");
}

function spansHTML(spans){
  if(!spans || !spans.length) return '<span class="spans">no spans</span>';
  return '<div class="spans">'+spans.map(sp=>{
    const cls = sp.is_name ? "" : "noname";
    return `<code class="${cls}">[${sp.start},${sp.end}] ${esc(sp.text)}${sp.is_name?"":" ·not-name"}</code>`;
  }).join(" ")+'</div>';
}

function cardHTML(rec){
  const raw = rec.raw||{};
  const dec = decisionOf(rec);
  const note = (rec.review||{}).note || "";
  const updated = (rec.review||{}).updated || "";
  const badges = [];
  badges.push(rec.valid ? '<span class="badge ok">valid</span>' : '<span class="badge err">INVALID</span>');
  if(dec==="approved") badges.push('<span class="badge dec-approved">✓ approved</span>');
  if(dec==="denied") badges.push('<span class="badge dec-denied">✗ denied</span>');
  if(raw.category) badges.push(`<span class="badge cat">${esc(raw.category)}</span>`);
  if(raw.register) badges.push(`<span class="chip">${esc(raw.register)}</span>`);
  if(raw.source) badges.push(`<span class="chip">${esc(raw.source)}</span>`);
  if(raw.quarantine) badges.push('<span class="badge q">quarantine</span>');
  if(raw.ambiguous_token) badges.push(`<span class="chip">tok: ${esc(raw.ambiguous_token)}</span>`);

  let body;
  if(VIEW==="rendered"){
    body = `<div class="lbl">target (names highlighted)</div>
            <div class="text rendered">${segHTML(rec.segments)}</div>`;
  } else {
    body = `<div class="lbl">input</div><div class="text">${esc(raw.input)}</div>
            <div class="lbl">target (raw tags)</div><div class="text">${esc(raw.target)}</div>`;
  }
  const err = rec.error ? `<div class="err-msg">⚠ ${esc(rec.error)}</div>` : "";
  const k = esc(rec.key);
  const review = `<div class="review-row">
      <button class="rbtn approve ${dec==="approved"?"active":""}" data-act="approve" data-key="${k}">✓ Approve</button>
      <button class="rbtn deny ${dec==="denied"?"active":""}" data-act="deny" data-key="${k}">✗ Deny</button>
      <button class="rbtn clear" data-act="clear" data-key="${k}">clear</button>
      <input class="note" type="text" placeholder="note (optional)…" value="${esc(note)}" data-key="${k}">
      <span class="updated">${updated?("saved "+esc(updated)):""}</span>
    </div>`;
  const cls = "card "+(rec.valid?"":"invalid ")+(dec?dec:"");
  return `<div class="${cls}" data-key="${k}">
    <div class="card-head"><span class="id">#${rec.line} · ${esc(raw.id||"(no id)")}</span> ${badges.join(" ")}</div>
    ${body}
    ${spansHTML(raw.spans)}
    ${err}
    ${review}
  </div>`;
}

function cardEl(key){ return document.querySelector('.card[data-key="'+(window.CSS&&CSS.escape?CSS.escape(key):key)+'"]'); }

function render(){
  if(!DATA || !DATA.records){ return; }
  FILTERED = DATA.records.filter(passFilters);
  const reviewed = DATA.stats.approved+DATA.stats.denied;
  $("#count").textContent = `${FILTERED.length} of ${DATA.records.length} shown · ${reviewed}/${DATA.stats.total} reviewed`;
  $("#list").innerHTML = FILTERED.length
    ? FILTERED.map(cardHTML).join("")
    : '<div class="empty">No rows match the current filters.</div>';
  if(FOCUS>=FILTERED.length) FOCUS = Math.max(0, FILTERED.length-1);
  applyFocus(false);
}

function applyFocus(scroll){
  document.querySelectorAll(".card.focused").forEach(c=>c.classList.remove("focused"));
  const rec = FILTERED[FOCUS];
  if(!rec) return;
  const el = cardEl(rec.key);
  if(el){ el.classList.add("focused"); if(scroll) el.scrollIntoView({block:"nearest", behavior:"smooth"}); }
}

async function setDecision(key, decision){
  const rec = DATA.records.find(r=>r.key===key);
  if(!rec) return;
  const note = (rec.review||{}).note || "";
  const res = await post("/api/review", {file:FILE, key, decision, note});
  if(res.error){ toast("save failed: "+res.error); return; }
  rec.review = res.review; // null when cleared
  recomputeReviewStats();
  renderStats();
  updateCard(rec);
  // If a decision filter is active and this row no longer matches, re-render the list.
  const show=$("#show").value;
  if((show==="pending"||show==="approved"||show==="denied") && !passFilters(rec)){
    const curKey = (FILTERED[FOCUS]||{}).key;
    render();
    // keep focus near where we were
    const idx = FILTERED.findIndex(r=>r.key===curKey);
    FOCUS = idx>=0 ? idx : Math.min(FOCUS, FILTERED.length-1);
    applyFocus(true);
  }
}

function updateCard(rec){
  const el = cardEl(rec.key);
  if(!el) return;
  const dec = decisionOf(rec);
  el.classList.remove("approved","denied");
  if(dec) el.classList.add(dec);
  el.querySelector('[data-act="approve"]').classList.toggle("active", dec==="approved");
  el.querySelector('[data-act="deny"]').classList.toggle("active", dec==="denied");
  // refresh badges + saved-time without full re-render
  const head = el.querySelector(".card-head");
  const idHtml = head.querySelector(".id").outerHTML;
  const raw = rec.raw||{};
  const badges = [];
  badges.push(rec.valid ? '<span class="badge ok">valid</span>' : '<span class="badge err">INVALID</span>');
  if(dec==="approved") badges.push('<span class="badge dec-approved">✓ approved</span>');
  if(dec==="denied") badges.push('<span class="badge dec-denied">✗ denied</span>');
  if(raw.category) badges.push(`<span class="badge cat">${esc(raw.category)}</span>`);
  if(raw.register) badges.push(`<span class="chip">${esc(raw.register)}</span>`);
  if(raw.source) badges.push(`<span class="chip">${esc(raw.source)}</span>`);
  if(raw.quarantine) badges.push('<span class="badge q">quarantine</span>');
  if(raw.ambiguous_token) badges.push(`<span class="chip">tok: ${esc(raw.ambiguous_token)}</span>`);
  head.innerHTML = idHtml + " " + badges.join(" ");
  const upd = el.querySelector(".updated");
  if(upd) upd.textContent = (rec.review||{}).updated ? ("saved "+(rec.review.updated)) : "";
}

async function saveNote(key, note){
  const rec = DATA.records.find(r=>r.key===key);
  if(!rec) return;
  const decision = decisionOf(rec);
  const res = await post("/api/review", {file:FILE, key, decision, note});
  if(!res.error){ rec.review = res.review; const upd=cardEl(key)?.querySelector(".updated"); if(upd) upd.textContent=(rec.review||{}).updated?("saved "+rec.review.updated):""; }
}

// event delegation for review buttons + notes
$("#list").addEventListener("click", e=>{
  const btn = e.target.closest(".rbtn");
  if(!btn) return;
  const key = btn.dataset.key, act = btn.dataset.act;
  // set focus to the clicked card
  const idx = FILTERED.findIndex(r=>r.key===key); if(idx>=0) { FOCUS=idx; applyFocus(false); }
  if(act==="approve") setDecision(key, decisionOf(DATA.records.find(r=>r.key===key))==="approved"?"":"approved");
  else if(act==="deny") setDecision(key, decisionOf(DATA.records.find(r=>r.key===key))==="denied"?"":"denied");
  else if(act==="clear") setDecision(key, "");
});
$("#list").addEventListener("change", e=>{
  if(e.target.classList.contains("note")) saveNote(e.target.dataset.key, e.target.value);
});

// seal / export
$("#seal").addEventListener("click", async ()=>{
  if(!FILE) return;
  const res = await post("/api/export", {file:FILE});
  if(res.error){ toast("export failed: "+res.error); return; }
  let msg = `🔏 Sealed ${res.written} approved row(s) → ${res.path}`;
  if(res.skipped_invalid) msg += ` (skipped ${res.skipped_invalid} invalid)`;
  toast(msg);
});

// keyboard shortcuts
document.addEventListener("keydown", e=>{
  if(/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)){
    if(e.key==="Escape") document.activeElement.blur();
    return;
  }
  const rec = FILTERED[FOCUS];
  if(e.key==="j"||e.key==="ArrowDown"){ e.preventDefault(); FOCUS=Math.min(FOCUS+1, FILTERED.length-1); applyFocus(true); }
  else if(e.key==="k"||e.key==="ArrowUp"){ e.preventDefault(); FOCUS=Math.max(FOCUS-1,0); applyFocus(true); }
  else if(rec && e.key==="a"){ e.preventDefault(); setDecision(rec.key, decisionOf(rec)==="approved"?"":"approved"); }
  else if(rec && e.key==="d"){ e.preventDefault(); setDecision(rec.key, decisionOf(rec)==="denied"?"":"denied"); }
  else if(rec && e.key==="u"){ e.preventDefault(); setDecision(rec.key, ""); }
  else if(rec && e.key==="n"){ e.preventDefault(); const el=cardEl(rec.key); const inp=el&&el.querySelector(".note"); if(inp) inp.focus(); }
});

// wiring
$("#file").addEventListener("change", e=>loadFile(e.target.value));
["#cat","#reg","#src","#show"].forEach(s=>$(s).addEventListener("change", ()=>{FOCUS=0; render();}));
$("#search").addEventListener("input", ()=>{ clearTimeout(window._t); window._t=setTimeout(()=>{FOCUS=0; render();},120); });
$("#viewRendered").addEventListener("click", ()=>{VIEW="rendered"; $("#viewRendered").classList.add("on"); $("#viewDiff").classList.remove("on"); render();});
$("#viewDiff").addEventListener("click", ()=>{VIEW="diff"; $("#viewDiff").classList.add("on"); $("#viewRendered").classList.remove("on"); render();});

loadFiles();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open a browser")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    files = discover_files()
    print(f"De-id test-set review UI -> {url}")
    print(f"  serving {len(files)} split(s):")
    for f in files:
        print(f"    {f['path']}  ({f['count']} rows)")
    print(f"  approvals persisted under: {REVIEWS_DIR.relative_to(REPO)}/")
    print("  Ctrl-C to stop.")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye.")
        server.shutdown()


if __name__ == "__main__":
    main()
