/* =====================================================================
   ML notes · shared runtime for programming/00-13
   theme, contents, progress, highlighting, maths, practice questions,
   and Python cells that run real scikit-learn in the browser (Pyodide).
   ===================================================================== */
(() => {
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const store = {
  get(k, d) { try { const v = JSON.parse(localStorage.getItem(k)); return v ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const ASSETS = (document.currentScript?.src || "").replace(/notes\.js.*$/, "");

/* ------------------------------------------------------------------
   Syntax highlighter (tiny sticky-regex tokenizer)
   ------------------------------------------------------------------ */
const RULES = {
  python: [
    [/#[^\n]*/y, "com"],
    [/[rbfu]{0,2}("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')/yi, "str"],
    [/\b\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?j?\b/yi, "num"],
    [/\b(?:def|class|return|if|elif|else|for|while|in|not|and|or|is|import|from|as|with|try|except|finally|raise|lambda|yield|async|await|pass|break|continue|global|nonlocal|del|assert|None|True|False)\b/y, "kw"],
    [/@[\w.]+/y, "attr"],
    [/[A-Za-z_]\w*(?=\s*\()/y, "fn"],
    [/[A-Za-z_]\w*/y, null],
  ],
  bash: [[/#[^\n]*/y, "com"], [/"[^"]*"|'[^']*'/y, "str"], [/(?<=^|\n|&& |\| )\s*[\w./-]+/y, "cmd"]],
  json: [[/"(?:\\.|[^"\\])*"(?=\s*:)/y, "attr"], [/"(?:\\.|[^"\\])*"/y, "str"], [/-?\b\d+(?:\.\d+)?\b/y, "num"], [/\b(?:true|false|null)\b/y, "kw"]],
  sql: [[/--[^\n]*/y, "com"], [/'(?:''|[^'])*'/y, "str"], [/\b\d+(?:\.\d+)?\b/y, "num"],
        [/\b(?:SELECT|FROM|WHERE|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|AND|OR|NOT|NULL|ORDER|BY|LIMIT|AS|JOIN|GROUP|COUNT)\b/yi, "kw"], [/[A-Za-z_]\w*/y, null]],
  text: [],
};
RULES.py = RULES.python; RULES.sh = RULES.shell = RULES.bash; RULES.txt = RULES.console = RULES.text;
function tokenize(src, rules) {
  let out = "", i = 0;
  outer: while (i < src.length) {
    for (const [re, cls] of rules) {
      re.lastIndex = i;
      const m = re.exec(src);
      if (m && m[0].length) { out += cls ? `<span class="t-${cls}">${esc(m[0])}</span>` : esc(m[0]); i += m[0].length; continue outer; }
    }
    out += esc(src[i]); i++;
  }
  return out;
}
const highlight = (src, lang) => tokenize(src, RULES[lang] || []);

function headFor(wrap, html) {
  const head = document.createElement("div");
  head.className = "codehead";
  head.innerHTML = html + `<span class="sp"></span>`;
  wrap.prepend(head);
  return head;
}
function copyBtn(head, getText) {
  const b = document.createElement("button");
  b.textContent = "Copy";
  b.onclick = () => navigator.clipboard?.writeText(getText()).then(() => { b.textContent = "Copied ✓"; setTimeout(() => b.textContent = "Copy", 1200); });
  head.append(b);
}
function enhanceCode(pre) {
  if (pre.dataset.done) return;
  pre.dataset.done = 1;
  const raw = pre.textContent.replace(/\n$/, "");
  pre.dataset.raw = raw;
  pre.innerHTML = highlight(raw, pre.dataset.lang || "text");
  const wrap = pre.closest(".codewrap");
  if (!wrap || wrap.hasAttribute("data-nohead") || wrap.classList.contains("pyc")) return;
  copyBtn(headFor(wrap, `<span>${esc(wrap.dataset.label || pre.dataset.lang || "text")}</span>`), () => raw);
}

/* ------------------------------------------------------------------
   Maths: <span class="tex">…</span> and <div class="tex-d">…</div>, rendered by KaTeX (vendored)
   ------------------------------------------------------------------ */
let katexLoad = null;
function renderMath(root = document) {
  const items = $$(".tex:not([data-r]), .tex-d:not([data-r])", root);
  if (!items.length) return;
  katexLoad = katexLoad || new Promise((res, rej) => {
    if (window.katex) return res();
    const css = document.createElement("link"); css.rel = "stylesheet"; css.href = ASSETS + "vendor/katex/katex.min.css"; document.head.append(css);
    const s = document.createElement("script"); s.src = ASSETS + "vendor/katex/katex.min.js"; s.onload = res; s.onerror = rej; document.head.append(s);
  });
  katexLoad.then(() => items.forEach(el => {
    el.dataset.r = 1;
    const src = el.textContent;
    try { window.katex.render(src, el, { displayMode: el.classList.contains("tex-d"), throwOnError: false, strict: "ignore", trust: false }); }
    catch { el.innerHTML = `<span class="tex-src">${esc(src)}</span>`; }
  })).catch(() => items.forEach(el => el.classList.add("tex-src")));
}

/* ------------------------------------------------------------------
   Python cells:
   <div class="codewrap pyc" [data-norun="why"] [data-dl="what it downloads"]>
     <pre class="code" data-lang="python">…</pre>
     <div class="outblk rec"><div class="oh">Recorded output</div> … rows: pre / .df / .plot … </div>
   </div>
   ▶ Run executes real CPython compiled to WebAssembly (Pyodide). Cells share one
   namespace in page order, like a notebook: running a cell first runs the cells
   above it that haven't run yet. numpy / pandas / scikit-learn / matplotlib / scipy
   load on demand; matplotlib figures and DataFrames show the way Jupyter shows them.
   <body data-py-data="california_housing"> serves that dataset from assets/data
   (the browser can't download it from scikit-learn's servers).
   ------------------------------------------------------------------ */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
let pyReady = null, pyBusy = Promise.resolve();
const PY_KERNEL = String.raw`
import ast as _ast, sys as _sys, io as _io, base64 as _b64, json as _json, traceback as _tb, os as _os
_os.environ.setdefault("MPLBACKEND", "agg")
import warnings as _w
_w.filterwarnings("ignore", category=FutureWarning)
_w.formatwarning = lambda m, c, f, l, line=None: f"{c.__name__}: {m}\n"     # no library paths in the output
class _MLOut:
    def __init__(self, kind): self.kind = kind
    def write(self, s):
        if not s: return 0
        o = _ML["out"]
        if o and o[-1][0] == self.kind: o[-1][1] += s
        else: o.append([self.kind, s])
        return len(s)
    def flush(self): pass
    def isatty(self): return False
_ML = {"out": []}
def _ml_show(v):
    if v is None: return
    h = getattr(v, "_repr_html_", None) if type(v).__module__.startswith("pandas") else None   # tables as HTML; estimators etc. as text
    if callable(h):
        try:
            s = h()
            if s: _ML["out"].append(["html", s]); return
        except Exception: pass
    _ML["out"].append(["text", repr(v) + "\n"])
def _ml_figs():
    if "matplotlib.pyplot" not in _sys.modules: return
    import matplotlib.pyplot as plt
    for n in plt.get_fignums():
        b = _io.BytesIO(); plt.figure(n).savefig(b, format="png", bbox_inches="tight", dpi=96)
        _ML["out"].append(["png", _b64.b64encode(b.getvalue()).decode()])
    plt.close("all")
def _ml_patch_show():
    if "matplotlib.pyplot" in _sys.modules:
        import matplotlib.pyplot as plt
        if getattr(plt.show, "_ml", False): return
        def show(*a, **k): _ml_figs()
        show._ml = True; plt.show = show
_ML_URLS = {}
def _ml_patch_pandas():
    import pandas as pd
    if getattr(pd.read_csv, "_ml", False): return
    orig = pd.read_csv
    def read_csv(path, *a, **k):
        if isinstance(path, str) and path in _ML_URLS: path = _ML_URLS[path]   # served from the notes' own copy
        return orig(path, *a, **k)
    read_csv._ml = True; pd.read_csv = read_csv
_ML_NS = {"__name__": "__main__"}
def _ml_run(src):
    _ML["out"] = []
    so, se = _sys.stdout, _sys.stderr
    _sys.stdout, _sys.stderr = _MLOut("text"), _MLOut("err")
    try:
        tree = _ast.parse(src, "<cell>", "exec")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], _ast.Expr) else None
        exec(compile(tree, "<cell>", "exec"), _ML_NS)
        _ml_patch_show()
        if last is not None:
            v = eval(compile(_ast.Expression(last.value), "<cell>", "eval"), _ML_NS)
            if not src.rstrip().endswith(";"): _ml_show(v)
        _ml_patch_show(); _ml_figs()
    except BaseException as e:
        lines = _tb.format_exception(type(e), e, e.__traceback__)
        keep = [l for l in lines if 'File "<exec>"' not in l and "/lib/python" not in l]
        _ML["out"].append(["err", "".join(keep[-6:])])
        try: _ml_figs()
        except Exception: pass
    finally:
        _sys.stdout, _sys.stderr = so, se
    return _json.dumps(_ML["out"])
`;
// datasets the page serves itself, wired into sklearn.datasets
const PY_DATA = {
  california_housing: String.raw`
def _ml_install_california(csv_gz, descr):
    import gzip, io, numpy as np, pandas as pd, sklearn.datasets as _d
    from sklearn.utils import Bunch
    frame = pd.read_csv(io.StringIO(gzip.decompress(csv_gz).decode()), float_precision="round_trip")
    names = list(frame.columns[:-1])
    def fetch_california_housing(*, data_home=None, download_if_missing=True, return_X_y=False, as_frame=False, n_retries=3, delay=1.0):
        X, y = frame[names].copy(), frame["MedHouseVal"].copy()
        if not as_frame: X, y = X.to_numpy(), y.to_numpy()
        if return_X_y: return X, y
        return Bunch(data=X, target=y, frame=frame.copy() if as_frame else None, target_names=["MedHouseVal"], feature_names=names, DESCR=descr)
    _d.fetch_california_housing = fetch_california_housing
    import sklearn.datasets._california_housing as _m; _m.fetch_california_housing = fetch_california_housing
`,
};
function loadPython(status) {
  if (pyReady) return pyReady;
  pyReady = (async () => {
    status("downloading Python (≈ 10 MB, only the first time)…");
    if (!window.loadPyodide) await new Promise((res, rej) => { const s = document.createElement("script"); s.src = PYODIDE + "pyodide.js"; s.onload = res; s.onerror = () => rej(new Error("could not load Pyodide (are you online?)")); document.head.append(s); });
    const py = await window.loadPyodide({ indexURL: PYODIDE });
    py.runPython(PY_KERNEL);
    for (const name of (document.body.dataset.pyData || "").split(",").filter(Boolean)) {
      status(`loading the ${name.replace(/_/g, " ")} data…`);
      await py.loadPackage(["numpy", "pandas", "scikit-learn"]);
      const [gz, descr] = await Promise.all([fetch(ASSETS + `data/${name}.csv.gz`).then(r => r.arrayBuffer()), fetch(ASSETS + `data/${name}.descr.txt`).then(r => r.text())]);
      py.runPython(PY_DATA[name]);
      py.globals.get(`_ml_install_${name.split("_")[0]}`)(py.toPy(new Uint8Array(gz)), descr);
    }
    return py;
  })();
  pyReady.catch(() => { pyReady = null; });
  return pyReady;
}
const PIP_ONLY = { seaborn: "seaborn" };      // not in the Pyodide distribution: install from PyPI
async function loadImports(py, code, status) {
  const want = new Set();
  for (const m of code.matchAll(/^\s*(?:from|import)\s+([A-Za-z_]\w*)/gm)) want.add(m[1]);
  const pip = [...want].filter(m => PIP_ONLY[m] && !py.runPython(`__import__("importlib.util").util.find_spec(${JSON.stringify(m)}) is not None`));
  status("loading " + ([...want].filter(m => !["sys", "os", "math", "random", "time", "itertools", "collections", "json", "re"].includes(m)).join(", ") || "Python") + "…");
  await py.loadPackagesFromImports(code);
  if (/\.(plot|hist|boxplot|scatter)\(|matplotlib|plt\.|seaborn|sns\./.test(code)) await py.loadPackage("matplotlib");
  if (pip.length) { await py.loadPackage("micropip"); await py.pyimport("micropip").install(pip.map(m => PIP_ONLY[m])); }
}
// small datasets the notebooks read from the internet are kept in assets/data (urls.json maps URL → file)
let urlMap = null;
const served = new Set();
async function serveUrls(py, code) {
  if (!/read_csv/.test(code)) return;
  urlMap = urlMap || await fetch(ASSETS + "data/urls.json").then(r => r.json()).catch(() => ({}));
  const want = Object.keys(urlMap).filter(u => code.includes(u) && !served.has(u));
  if (!want.length) return;
  await py.loadPackage("pandas");
  py.FS.mkdirTree("/data/remote");
  for (const u of want) {
    const buf = new Uint8Array(await fetch(ASSETS + "data/" + urlMap[u]).then(r => r.arrayBuffer()));
    const path = "/data/" + urlMap[u];
    py.FS.writeFile(path, buf);
    py.runPython(`_ML_URLS[${JSON.stringify(u)}] = ${JSON.stringify(path)}`);
    served.add(u);
  }
  py.runPython("_ml_patch_pandas()");
}
function renderRows(box, rows) {
  box.innerHTML = "";
  for (const [kind, val] of rows) {
    const row = document.createElement("div"); row.className = "orow";
    if (kind === "png") { row.classList.add("plot"); const img = new Image(); img.src = "data:image/png;base64," + val; img.onclick = () => img.classList.toggle("big"); row.append(img); }
    else if (kind === "html") { row.classList.add("df"); row.innerHTML = val; $$("style, script", row).forEach(x => x.remove()); }
    else { const pre = document.createElement("pre"); if (kind === "err") pre.className = "err"; pre.textContent = val.replace(/\n$/, ""); row.append(pre); }
    box.append(row);
  }
  if (!rows.length) box.innerHTML = `<div class="orow"><pre><span class="inf">(ran, showed nothing)</span></pre></div>`;
}
const pyCells = () => $$(".codewrap.pyc:not([data-norun])");
const cellCode = w => $("textarea.pyedit", w)?.value ?? $("pre.code", w).dataset.raw;
function enhancePyCell(wrap) {
  const pre = $("pre.code", wrap);
  enhanceCode(pre);
  const norun = wrap.dataset.norun, dl = wrap.dataset.dl;
  const badge = norun ? `<span class="badge" title="${esc(norun)}">recorded output only</span>`
    : dl ? `<span class="badge dl" title="${esc(dl)}">downloads data</span>` : `<span class="badge pyb" title="Runs real Python">runs Python</span>`;
  const head = headFor(wrap, `<span class="file">🐍 <b>Python</b></span>` + badge);
  if (!norun) {
    const run = document.createElement("button"); run.className = "run"; run.textContent = "▶ Run";
    run.onclick = () => runPyCell(wrap); head.append(run);
    const ed = document.createElement("button"); ed.textContent = "Edit"; ed.className = "edit";
    ed.onclick = () => {
      let ta = $("textarea.pyedit", wrap);
      if (ta) { pre.textContent = ta.value; pre.dataset.done = ""; enhanceCode(pre); ta.remove(); pre.hidden = false; ed.textContent = "Edit"; return; }
      ta = document.createElement("textarea"); ta.className = "ed pyedit"; ta.spellcheck = false; ta.value = pre.dataset.raw;
      ta.style.cssText = "width:100%;border:0;border-radius:0;font:13.5px/1.6 var(--mono);padding:14px 16px;background:var(--code-bg);color:var(--text);min-height:" + Math.min(520, pre.offsetHeight + 20) + "px";
      ta.onkeydown = e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey || e.shiftKey)) { e.preventDefault(); runPyCell(wrap); } };
      pre.hidden = true; pre.after(ta); ta.focus(); ed.textContent = "Done";
    };
    head.append(ed);
    const add = document.createElement("button"); add.textContent = "+ cell"; add.title = "Add an empty cell below for your own code";
    add.onclick = () => addScratchCell(wrap); head.append(add);
  }
  copyBtn(head, () => cellCode(wrap));
  if (norun) { const n = document.createElement("div"); n.className = "pynote"; n.textContent = "⚙ " + norun; wrap.append(n); }
  // long recorded text outputs fold
  $$(".outblk.rec pre", wrap).forEach(p => {
    if (p.textContent.split("\n").length <= 16) return;
    p.classList.add("clip");
    const b = document.createElement("button"); b.className = "more"; b.textContent = "Show all ▾";
    b.onclick = () => { const c = p.classList.toggle("clip"); b.textContent = c ? "Show all ▾" : "Show less ▴"; };
    p.after(b);
  });
  $$(".outblk .plot img", wrap).forEach(img => img.onclick = () => img.classList.toggle("big"));
}

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

async function runPyCell(wrap) {
  if ((await liveStatus()).enabled) return runLiveCell(wrap);
  const cells = pyCells(), idx = cells.indexOf(wrap);
  let live = $(".outblk.live", wrap);
  if (!live) { live = document.createElement("div"); live.className = "outblk live"; live.innerHTML = `<div class="oh">Your browser · Python</div><div class="rows"></div>`; wrap.append(live); }
  const box = $(".rows", live);
  const status = s => { box.innerHTML = `<div class="orow"><pre><span class="inf">${esc(s)}</span></pre></div>`; };
  status("starting…");
  pyBusy = pyBusy.then(async () => {
    let py;
    try { py = await loadPython(status); } catch (e) { box.innerHTML = `<div class="orow"><pre class="err">${esc(e.message)}</pre></div>`; return; }
    const run = async (w, show) => {
      const code = cellCode(w);
      try { await loadImports(py, code, show ? status : () => {}); await serveUrls(py, code); } catch (e) { return [["err", "could not load a package: " + e.message]]; }
      return JSON.parse(py.globals.get("_ml_run")(code));
    };
    for (let i = 0; i < idx; i++) {
      if (cells[i].dataset.ran) continue;
      status(`running cell ${i + 1} of ${idx} first (the cells above set things up)…`);
      await run(cells[i], false); cells[i].dataset.ran = 1;
    }
    status("running…");
    const rows = await run(wrap, true);
    wrap.dataset.ran = 1;
    renderRows(box, rows);
  });
  await pyBusy;
}

/* Tab set: <div class="tabset"><div class="tabpane" data-label="…">…</div>…</div> */
function enhanceTabs(ts) {
  const panes = $$(":scope > .tabpane", ts);
  const bar = document.createElement("div"); bar.className = "tabbar";
  panes.forEach((p, i) => {
    const b = document.createElement("button"); b.textContent = p.dataset.label;
    b.onclick = () => { panes.forEach((q, k) => q.hidden = k !== i); $$("button", bar).forEach((x, k) => x.classList.toggle("on", k === i)); };
    bar.append(b);
  });
  ts.prepend(bar);
  bar.firstChild && bar.firstChild.click();
}

/* ------------------------------------------------------------------
   Practice questions (.pq): only questions the author adds by hand
   <div class="pq" data-answer="A, C" data-type="single|multi|num" [data-tol] [data-tag]>
     <div class="pq-q">…</div><ol class="pq-opts"><li>…<span class="pq-note">…</span></li></ol><div class="pq-why">…</div>
   ------------------------------------------------------------------ */
function initQuestions() {
  const results = new Map(), all = $$(".pq");
  const tally = () => {
    const el = $("#pqTally"); if (!el) return;
    el.parentElement.hidden = !all.length;
    el.textContent = `${[...results.values()].filter(Boolean).length} / ${all.length}`;
  };
  all.forEach((el, n) => {
    const type = el.dataset.type || "single"; el.dataset.type = type;
    const answer = type === "text" ? el.dataset.answer.split("|").map(s => s.trim()) : el.dataset.answer.split(",").map(s => s.trim().toUpperCase());
    const qEl = $(".pq-q", el), ol = $(".pq-opts", el), why = $(".pq-why", el);
    const head = document.createElement("div"); head.className = "pq-head";
    head.innerHTML = `<span class="pq-status"><span class="qn">Q${n + 1}</span><span class="st">Question</span></span>` +
      (el.dataset.tag ? `<span class="pq-tag">${esc(el.dataset.tag)}</span>` : "") +
      `<span class="pq-kind">${{ single: "single choice", multi: "select all that apply", num: "numeric answer", text: "type the answer" }[type]}</span><span class="pq-sc"></span>`;
    const body = document.createElement("div"); body.className = "pq-body";
    [qEl, ol].forEach(x => x && body.append(x));
    const items = ol ? [...ol.children] : [];
    items.forEach((li, i) => {
      const k = String.fromCharCode(65 + i); li.dataset.k = k; li.tabIndex = 0;
      const t = document.createElement("div"); t.className = "pq-text";
      while (li.firstChild) t.append(li.firstChild);
      li.innerHTML = `<span class="pq-mark"></span><span class="pq-letter">${k}.</span>`; li.append(t);
    });
    let input;
    if (type === "num" || type === "text") {
      const b = document.createElement("div"); b.className = "pq-num";
      b.innerHTML = `<input type="text" inputmode="${type === "num" ? "decimal" : "text"}" placeholder="Your answer"><span class="exp"></span>`;
      body.append(b); input = $("input", b);
    }
    const act = document.createElement("div"); act.className = "pq-actions";
    act.innerHTML = `<button class="btn primary pq-check">Check answer</button><button class="btn pq-retry" hidden>↺ Try again</button><button class="btn pq-show" title="See the answer and the solution without answering (not scored)">Show answer</button>`;
    body.append(act);
    const check = $(".pq-check", act), retry = $(".pq-retry", act), show = $(".pq-show", act);
    if (type === "single") check.hidden = true;
    if (why) body.append(why);
    el.prepend(head); el.append(body);
    const grade = chosen => {
      let ok;
      if (type === "num") {
        const v = parseFloat(String(input.value).replace(/,/g, "")); if (Number.isNaN(v)) return;
        ok = Math.abs(v - parseFloat(answer[0])) <= parseFloat(el.dataset.tol || "0");
        input.disabled = true; $(".exp", body).innerHTML = ok ? "" : `Expected: <b>${esc(el.dataset.answer)}</b>`;
      } else if (type === "text") {
        const norm = s => String(s).trim().replace(/\s+/g, " ").toLowerCase(); if (!norm(input.value)) return;
        ok = answer.some(a => norm(a) === norm(input.value));
        input.disabled = true; $(".exp", body).innerHTML = ok ? "" : `Expected: <b>${esc(answer.join(" / "))}</b>`;
      } else {
        if (!chosen.length) return;
        ok = chosen.length === answer.length && chosen.every(x => answer.includes(x));
        items.forEach(li => {
          const k = li.dataset.k, p = chosen.includes(k), c = answer.includes(k);
          if (p && c) { li.classList.add("b-right"); li.dataset.badge = "Correct"; }
          else if (p) { li.classList.add("b-wrong"); li.dataset.badge = "Wrong"; }
          else if (c) { li.classList.add("b-exp"); li.dataset.badge = type === "multi" ? "Missed" : "Expected"; }
        });
      }
      el.classList.add("done", ok ? "right" : "wrong");
      $(".st", head).textContent = ok ? "Answer is Correct ✓" : "Answer is Wrong";
      $(".pq-sc", head).textContent = `Score: ${ok ? 1 : 0} / 1`;
      check.hidden = true; show.hidden = true; retry.hidden = false; results.set(el, ok); tally();
    };
    show.onclick = () => {
      if (el.classList.contains("done")) return;
      if (type === "num" || type === "text") { input.disabled = true; $(".exp", body).innerHTML = `Answer: <b>${esc(type === "text" ? answer.join(" / ") : el.dataset.answer)}</b>`; }
      else items.forEach(li => { if (answer.includes(li.dataset.k)) { li.classList.add("b-exp"); li.dataset.badge = "Answer"; } });
      el.classList.add("done", "shown");
      $(".st", head).textContent = "Answer shown";
      check.hidden = show.hidden = true; retry.hidden = false; results.set(el, false); tally();
    };
    retry.onclick = () => {
      el.classList.remove("done", "right", "wrong", "shown"); $(".st", head).textContent = "Question"; $(".pq-sc", head).textContent = ""; show.hidden = false;
      items.forEach(li => { li.classList.remove("sel", "b-right", "b-wrong", "b-exp"); delete li.dataset.badge; });
      if (input) { input.disabled = false; input.value = ""; $(".exp", body).innerHTML = ""; }
      check.hidden = type === "single"; retry.hidden = true; results.delete(el); tally();
    };
    const pick = li => { if (el.classList.contains("done")) return; if (type === "single") { li.classList.add("sel"); grade([li.dataset.k]); } else li.classList.toggle("sel"); };
    items.forEach(li => { li.onclick = e => { if (!e.target.closest("a")) pick(li); }; li.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(li); } }; });
    check.onclick = () => grade(items.filter(li => li.classList.contains("sel")).map(li => li.dataset.k));
    if (input) input.onkeydown = e => { if (e.key === "Enter") check.click(); };
  });
  tally();
}

/* ------------------------------------------------------------------
   Page chrome: theme, progress, contents (with each section's questions), done tracking
   ------------------------------------------------------------------ */
function initChrome() {
  const root = document.documentElement;
  root.dataset.theme = store.get("ml-notes-theme", matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const tb = $("#themeBtn"); if (tb) tb.onclick = () => { root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark"; store.set("ml-notes-theme", root.dataset.theme); };
  const mb = $("#menuBtn"); if (mb) mb.onclick = () => $("#toc").classList.toggle("open");
  const pr = $("#progress");
  addEventListener("scroll", () => { const h = document.documentElement; if (pr) pr.style.width = (h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) * 100) + "%"; }, { passive: true });
  const rt = $("#readTime"); if (rt && $(".content")) rt.textContent = `⏱ ~${Math.max(1, Math.round($(".content").innerText.split(/\s+/).length / 170))} min`;
  $$("figure img").forEach(img => img.onclick = () => img.classList.toggle("big"));

  const nav = $("#tocNav"); if (!nav) return;
  const sections = $$("section.sec");
  for (const el of sections) {
    const a = document.createElement("a"); a.href = "#" + el.id; a.dataset.id = el.id;
    a.innerHTML = `<span class="n">${esc(el.dataset.num || "")}</span><span>${esc(el.dataset.title)}</span><span class="tick">✓</span>`;
    nav.append(a);
    $$("h3[id]", el).forEach(h => { const s = document.createElement("a"); s.href = "#" + h.id; s.className = "sub"; s.dataset.id = h.id; s.textContent = h.textContent; nav.append(s); });
    $$(".pq", el).forEach(q => {   // every question in the section gets its own line in the index
      const n = $$(".pq").indexOf(q) + 1; if (!q.id) q.id = "pq-" + n;
      const s = document.createElement("a"); s.href = "#" + q.id; s.className = "sub qlink"; s.dataset.id = q.id;
      const qq = $(".pq-q", q)?.cloneNode(true); if (qq) $$("pre, .codewrap, figure, svg, .tex-d", qq).forEach(x => x.replaceWith(" "));   // words only, not the code
      const label = (qq?.textContent || "").replace(/\s+/g, " ").trim().replace(/[,:]$/, "") || "Question";
      s.innerHTML = `<span class="qb">Q${n}</span><span class="qt">${esc(label)}</span>`; s.title = label; nav.append(s);
    });
  }
  nav.addEventListener("click", () => $("#toc").classList.remove("open"));
  const spy = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) $$("#tocNav a").forEach(a => a.classList.toggle("active", a.dataset.id === en.target.id)); }), { rootMargin: "-60px 0px -70% 0px" });
  $$("section.sec, section.sec h3[id], section.sec .pq[id]").forEach(t => spy.observe(t));

  const key = `ml-${document.body.dataset.subject}-${document.body.dataset.page}-done`;
  const done = new Set(store.get(key, []));
  const render = () => {
    $$(".donebtn").forEach(b => { const on = done.has(b.dataset.done); b.classList.toggle("done", on); b.textContent = on ? "✓ Understood" : "Mark as understood"; });
    $$("#tocNav a:not(.sub)").forEach(a => a.classList.toggle("done", done.has(a.dataset.id)));
    const c = $("#doneCount"); if (c) { c.textContent = [...done].filter(id => sections.some(s => s.id === id)).length; $("#doneTotal").textContent = sections.length; $("#doneBar").style.width = (c.textContent / Math.max(1, sections.length) * 100) + "%"; }
    store.set(`ml-${document.body.dataset.subject}-${document.body.dataset.page}-total`, sections.length);
  };
  $$(".donebtn").forEach(b => b.onclick = () => { done.has(b.dataset.done) ? done.delete(b.dataset.done) : done.add(b.dataset.done); store.set(key, [...done]); render(); });
  render();
}

/* index pages: progress per note from localStorage */
function initIndex() {
  $$("[data-progress]").forEach(card => {
    const [subj, page] = card.dataset.progress.split("/");
    const done = store.get(`ml-${subj}-${page}-done`, []).length, total = store.get(`ml-${subj}-${page}-total`, 0);
    const bar = $(".pb i", card), t = $(".pt", card);
    if (bar && total) bar.style.width = Math.min(100, done / total * 100) + "%";
    if (t) t.textContent = total ? `${Math.min(done, total)} / ${total} sections understood` : "not started";
  });
}

window.N = { $, $$, esc, store, highlight, enhanceCode, loadPython, renderMath };
document.addEventListener("DOMContentLoaded", () => {
  $$("pre.code").forEach(p => { if (!p.closest(".pyc")) enhanceCode(p); });
  $$(".codewrap.pyc").forEach(enhancePyCell);
  if ($$(".codewrap.pyc").length) liveStatus().then(s => {
    $$(".badge.pyb").forEach(b => { b.textContent = s.enabled ? "runs on this computer · PyTorch" : "runs in the browser"; b.title = s.enabled ? "Runs on the app's Python kernel: real PyTorch, with the GPU if there is one" : "Runs real Python (Pyodide) in this tab"; });
    if (s.enabled) { kernelChip(s.alive ? "idle" : "off"); const c = pyCells(); (s.ran || []).forEach(i => c[i] && (c[i].dataset.ran = 1)); }
  });
  $$(".tabset").forEach(enhanceTabs);
  initQuestions();
  initChrome();
  initIndex();
  renderMath();
});
})();
