"""One-off: give notes.js a live mode that runs cells on the app's Python kernel (PyTorch, GPU), with Pyodide as the fallback."""
import pathlib, sys
p = pathlib.Path(__file__).resolve().parents[3] / "notes/assets/notes.js"; t = p.read_text(encoding="utf-8")

LIVE = r'''
/* ------------------------------------------------------------------
   Live mode: inside Studyloom, cells run on a real Python kernel on this computer (PyTorch, the GPU),
   one kernel per note, so state carries from cell to cell like a notebook. Output streams in as the
   cell runs. Without the app (a note opened on its own), cells fall back to Pyodide in the browser.
   ------------------------------------------------------------------ */
const NOTE_ID = (location.pathname.match(/\/content\/(.+?)\.html$/) || [])[1] || "";
let liveInfo = null;
function liveStatus() {
  if (!liveInfo) liveInfo = (async () => {
    if (!NOTE_ID) return { enabled: false };
    try {
      const r = await fetch(`/api/kernel/status?note=${encodeURIComponent(NOTE_ID)}`, { credentials: "same-origin", headers: { "X-Studyloom": "1" } });
      return r.ok ? await r.json() : { enabled: false };
    } catch { return { enabled: false }; }
  })();
  return liveInfo;
}
let chip = null;
function kernelChip(state) {
  if (!chip) {
    chip = document.createElement("div"); chip.className = "kchip";
    chip.innerHTML = `<span class="dot"></span><span class="kt">Python kernel</span><button type="button" title="Start a fresh kernel: clears every variable">↺ Restart</button>`;
    $("button", chip).onclick = async () => { await restartKernel(); };
    document.body.append(chip);
  }
  chip.dataset.state = state;
  $(".kt", chip).textContent = { idle: "Python kernel · idle", busy: "Python kernel · running…", off: "Python kernel · not started" }[state] || "Python kernel";
}
async function restartKernel() {
  await fetch("/api/kernel/restart", { method: "POST", credentials: "same-origin", headers: { "X-Studyloom": "1", "Content-Type": "application/json" }, body: JSON.stringify({ note: NOTE_ID }) }).catch(() => {});
  pyCells().forEach(c => delete c.dataset.ran);
  kernelChip("off");
}
async function liveRun(code, cell, onEvent, signal) {
  const r = await fetch("/api/kernel/run", { method: "POST", credentials: "same-origin", signal,
    headers: { "X-Studyloom": "1", "Content-Type": "application/json" }, body: JSON.stringify({ note: NOTE_ID, cell, code }) });
  if (!r.ok) { const j = await r.json().catch(() => ({})); onEvent({ t: "err", d: (j.error || `The kernel refused the cell (${r.status}).`) + "\n" }); return false; }
  const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "", ok = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.done) ok = ev.ok; else onEvent(ev);
    }
  }
  return ok;
}
// streamed rows: text merges into the previous text block; \r redraws the current line (progress bars)
function liveSink(box) {
  box.innerHTML = "";
  let last = null, any = false;
  return ev => {
    if (ev.t === "info") return;
    any = true;
    if (ev.t === "text" || ev.t === "err") {
      if (!last || last.kind !== ev.t) {
        const row = document.createElement("div"); row.className = "orow";
        const pre = document.createElement("pre"); if (ev.t === "err") pre.className = "err";
        row.append(pre); box.append(row); last = { kind: ev.t, pre, text: "" };
      }
      for (const part of ev.d.split(/(\r)/)) {
        if (part === "\r") { const nl = last.text.lastIndexOf("\n"); last.text = last.text.slice(0, nl + 1); }
        else last.text += part;
      }
      last.pre.textContent = last.text.replace(/\n$/, "");
    } else { renderRowsAppend(box, [[ev.t, ev.d]]); last = null; }
    box.dataset.any = any ? "1" : "";
  };
}
function renderRowsAppend(box, rows) {
  const tmp = document.createElement("div"); renderRows(tmp, rows);
  while (tmp.firstChild) box.append(tmp.firstChild);
}
let liveBusy = Promise.resolve();
async function runLiveCell(wrap) {
  const cells = pyCells(), idx = cells.indexOf(wrap);
  let live = $(".outblk.live", wrap);
  if (!live) { live = document.createElement("div"); live.className = "outblk live"; live.innerHTML = `<div class="oh">This computer · Python kernel</div><div class="rows"></div>`; wrap.append(live); }
  const box = $(".rows", live), run = $("button.run", wrap);
  const status = s => { box.innerHTML = `<div class="orow"><pre><span class="inf">${esc(s)}</span></pre></div>`; };
  status("waiting for the kernel…");
  const ctl = new AbortController();
  const stop = document.createElement("button"); stop.className = "stop"; stop.textContent = "■ Stop";
  stop.onclick = async () => { ctl.abort(); await restartKernel(); status("stopped: the kernel was restarted, so variables from earlier cells are gone (run them again)."); };
  run.hidden = true; run.after(stop);
  liveBusy = liveBusy.then(async () => {
    kernelChip("busy");
    try {
      for (let i = 0; i < idx; i++) {
        if (cells[i].dataset.ran || cells[i].dataset.scratch) continue;
        status(`running cell ${i + 1} of ${idx} first (the cells above set things up)…`);
        const sink = liveSink(document.createElement("div"));
        if (!(await liveRun(cellCode(cells[i]), i, sink, ctl.signal))) { status(`cell ${i + 1} above failed; run it to see why.`); return; }
        cells[i].dataset.ran = 1;
      }
      status("running…");
      const sink = liveSink(box);
      const ok = await liveRun(cellCode(wrap), wrap.dataset.scratch ? -1 : idx, sink, ctl.signal);
      if (!box.dataset.any) box.innerHTML = `<div class="orow"><pre><span class="inf">(ran, showed nothing)</span></pre></div>`;
      if (ok) wrap.dataset.ran = 1;
    } catch (e) { if (e.name !== "AbortError") status("could not reach the kernel: " + e.message); }
    finally { stop.remove(); run.hidden = false; kernelChip("idle"); }
  });
  await liveBusy;
}
// "+ cell": an empty editable cell for your own experiments, sharing the note's kernel
function addScratchCell(after) {
  const w = document.createElement("div"); w.className = "codewrap pyc scratch"; w.dataset.scratch = "1";
  w.innerHTML = `<pre class="code" data-lang="python"># your own code: it shares variables with the cells above\n</pre>`;
  after.after(w); enhancePyCell(w);
  $("button.edit", w)?.click();
}
'''

pairs = [
    ("async function runPyCell(wrap) {", LIVE + "\nasync function runPyCell(wrap) {\n  if ((await liveStatus()).enabled) return runLiveCell(wrap);"),
    ("""    : dl ? `<span class="badge dl" title="${esc(dl)}">downloads data</span>` : `<span class="badge" title="Runs real Python (Pyodide) in this tab">runs in the browser</span>`;""",
     """    : dl ? `<span class="badge dl" title="${esc(dl)}">downloads data</span>` : `<span class="badge pyb" title="Runs real Python">runs Python</span>`;"""),
    ("""    const ed = document.createElement("button"); ed.textContent = "Edit";""",
     """    const ed = document.createElement("button"); ed.textContent = "Edit"; ed.className = "edit";"""),
    ("""    head.append(ed);
  }""", """    head.append(ed);
    const add = document.createElement("button"); add.textContent = "+ cell"; add.title = "Add an empty cell below for your own code";
    add.onclick = () => addScratchCell(wrap); head.append(add);
  }"""),
]
for a, b in pairs:
    if a not in t: sys.exit("not found: " + a[:80])
    t = t.replace(a, b, 1)
# the badge says where the cell will run, once we know
t = t.replace("""  $$(".codewrap.pyc").forEach(enhancePyCell);""", """  $$(".codewrap.pyc").forEach(enhancePyCell);
  if ($$(".codewrap.pyc").length) liveStatus().then(s => {
    $$(".badge.pyb").forEach(b => { b.textContent = s.enabled ? "runs on this computer · PyTorch" : "runs in the browser"; b.title = s.enabled ? "Runs on the app's Python kernel: real PyTorch, with the GPU if there is one" : "Runs real Python (Pyodide) in this tab"; });
    if (s.enabled) { kernelChip(s.alive ? "idle" : "off"); const c = pyCells(); (s.ran || []).forEach(i => c[i] && (c[i].dataset.ran = 1)); }
  });""", 1)
p.write_text(t, encoding="utf-8"); print("notes.js live mode added")

css = pathlib.Path(p.parent / "notes.css")
c = css.read_text(encoding="utf-8")
if ".kchip" not in c:
    c += """
/* live kernel */
.kchip{position:fixed;left:14px;bottom:14px;z-index:50;display:flex;align-items:center;gap:8px;padding:6px 8px 6px 12px;border:1px solid var(--line,#ddd);border-radius:999px;background:var(--panel,#fff);font:12.5px/1.2 var(--sans,system-ui);box-shadow:0 4px 18px rgba(0,0,0,.08)}
.kchip .dot{width:8px;height:8px;border-radius:50%;background:#9ca3af}
.kchip[data-state=idle] .dot{background:#16a34a}
.kchip[data-state=busy] .dot{background:#f59e0b;animation:kpulse 1s infinite}
.kchip button{border:0;background:transparent;cursor:pointer;font:inherit;color:inherit;padding:3px 8px;border-radius:999px}
.kchip button:hover{background:rgba(0,0,0,.06)}
@keyframes kpulse{50%{opacity:.35}}
.codewrap .stop{color:#b91c1c}
.codewrap.scratch{outline:1px dashed var(--line,#ccc);outline-offset:3px}
"""
    css.write_text(c, encoding="utf-8"); print("css added")
