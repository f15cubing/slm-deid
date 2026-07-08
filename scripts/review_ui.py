#!/usr/bin/env python3
"""Local web UI to review the de-id dataset splits.

Browse any ``*.jsonl`` split (train / val / hardcases), see the ``target`` rendered
with ⟨NAME⟩ tags highlighted, inspect gold spans + metadata, filter by
category/register/source, full-text search, and — the point of "make sure it's good" —
every record is run through :meth:`schema.Example.validate` so schema/integrity
violations are flagged right in the list.

    python scripts/review_ui.py            # serve http://127.0.0.1:8765
    python scripts/review_ui.py --port 9000
    python scripts/review_ui.py --no-open  # don't auto-open a browser

No third-party deps — stdlib ``http.server`` only.
"""

from __future__ import annotations

import argparse
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


def load_records(rel_path: str) -> dict:
    """Parse one split, validate each row, and return records + aggregate stats.

    Guards against path traversal: only files inside the repo are served.
    """
    target = (REPO / rel_path).resolve()
    if REPO not in target.parents:
        raise ValueError("path escapes repo")
    if not target.is_file():
        raise FileNotFoundError(rel_path)

    records: list[dict] = []
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "with_spans": 0,
        "quarantine": 0,
        "categories": {},
        "registers": {},
        "sources": {},
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
                # Server-side render of the tagged target into segments so the client
                # doesn't need to re-parse the ⟨NAME⟩ sentinels.
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
            records.append(rec)

    return {"path": rel_path, "records": records, "stats": stats}


def render_segments(target: str) -> list[dict]:
    """Split a tagged target into ``[{text, name}]`` segments for highlighting.

    ``name=True`` segments are the ⟨NAME⟩…⟨/NAME⟩ contents. Falls back to a single
    plain segment if the tags are malformed (so bad rows still display something).
    """
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
        if c == -1:  # shouldn't happen if well-formed, but be safe
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
            qs = parse_qs(parsed.query)
            rel = (qs.get("file") or [""])[0]
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
    --warn:#ffcc66;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; }
  body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .app { display:grid; grid-template-columns:300px 1fr; height:100vh; }
  .sidebar { background:var(--panel); border-right:1px solid var(--border); padding:16px; overflow:auto; }
  .main { overflow:auto; padding:20px 24px; }
  h1 { font-size:15px; margin:0 0 14px; letter-spacing:.02em; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:18px 0 8px; }
  select,input[type=text] { width:100%; background:var(--panel2); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:7px 9px; font:inherit; }
  label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  .chip { display:inline-block; padding:1px 7px; border-radius:999px; font-size:11px; background:var(--panel2); border:1px solid var(--border); color:var(--muted); margin:0 4px 4px 0; }
  .stat { display:flex; justify-content:space-between; padding:2px 0; font-size:12px; color:var(--muted); }
  .stat b { color:var(--fg); font-weight:600; }
  .stat.bad b { color:var(--bad); }
  .stat.ok b { color:var(--accent2); }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  .card.invalid { border-color:var(--bad); background:var(--bad-bg); }
  .card-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
  .id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--muted); }
  .badge { font-size:11px; padding:1px 8px; border-radius:6px; font-weight:600; }
  .badge.ok { background:#123; color:var(--accent2); border:1px solid #2c5; }
  .badge.err { background:#311; color:var(--bad); border:1px solid var(--bad); }
  .badge.q { background:#332; color:var(--warn); border:1px solid #664; }
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
  .toolbar { display:flex; gap:10px; align-items:center; margin-bottom:16px; flex-wrap:wrap; }
  .count { color:var(--muted); font-size:13px; }
  .empty { color:var(--muted); text-align:center; padding:60px 0; }
  .toggle { display:flex; gap:6px; }
  .btn { background:var(--panel2); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 12px; cursor:pointer; font:inherit; }
  .btn:hover { border-color:var(--accent); }
  .toggle .btn.on { border-color:var(--accent); color:var(--accent); }
  a { color:var(--accent); }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>De-id test-set review</h1>
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
      <option value="invalid">⚠ invalid only</option>
      <option value="spans">with name spans</option>
      <option value="nospans">no name spans (negatives)</option>
      <option value="quarantine">quarantined</option>
    </select>

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
    </div>
    <div id="list"></div>
  </main>
</div>

<script>
const $ = s => document.querySelector(s);
let DATA = null;      // {records, stats}
let VIEW = "rendered";

async function j(url){ const r = await fetch(url); return r.json(); }

function esc(s){ return (s??"").replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function loadFiles(){
  const files = await j("/api/files");
  const sel = $("#file");
  sel.innerHTML = files.map(f=>`<option value="${esc(f.path)}">${esc(f.path)} (${f.count})</option>`).join("");
  // Default to a val/test split if present.
  const pref = files.find(f=>/val|test/.test(f.path)) || files[0];
  if(pref){ sel.value = pref.path; await loadFile(pref.path); }
}

async function loadFile(path){
  DATA = await j("/api/records?file="+encodeURIComponent(path));
  if(DATA.error){ $("#list").innerHTML = `<div class="empty">${esc(DATA.error)}</div>`; return; }
  fillFilter("#cat", DATA.stats.categories);
  fillFilter("#reg", DATA.stats.registers);
  fillFilter("#src", DATA.stats.sources);
  renderStats();
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

function renderStats(){
  const s = DATA.stats;
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
  if(show==="invalid" && rec.valid) return false;
  if(show==="spans" && !hasName) return false;
  if(show==="nospans" && hasName) return false;
  if(show==="quarantine" && !raw.quarantine) return false;
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
  const badges = [];
  badges.push(rec.valid ? '<span class="badge ok">valid</span>' : '<span class="badge err">INVALID</span>');
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
  return `<div class="card ${rec.valid?"":"invalid"}">
    <div class="card-head"><span class="id">#${rec.line} · ${esc(raw.id||"(no id)")}</span> ${badges.join(" ")}</div>
    ${body}
    ${spansHTML(raw.spans)}
    ${err}
  </div>`;
}

function render(){
  if(!DATA || !DATA.records){ return; }
  const rows = DATA.records.filter(passFilters);
  $("#count").textContent = `${rows.length} of ${DATA.records.length} shown`;
  $("#list").innerHTML = rows.length
    ? rows.map(cardHTML).join("")
    : '<div class="empty">No rows match the current filters.</div>';
}

// wiring
$("#file").addEventListener("change", e=>loadFile(e.target.value));
["#cat","#reg","#src","#show"].forEach(s=>$(s).addEventListener("change", render));
$("#search").addEventListener("input", ()=>{ clearTimeout(window._t); window._t=setTimeout(render,120); });
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
