"""Build note pages from authored markdown, running every Python cell for real.

Usage:  python build.py <subject-src-dir> <notes-dir> [page-stem ...]   (output: <notes-dir>/<subject>, or $NOTES_OUT)

Authored page  (<subject-src-dir>/<NN_slug>.md):
    ---
    title: …            eyebrow: …        lead: one paragraph (markdown)
    badge: sk           data: california_housing (served to the browser)
    blurb: index text
    ---
    preface markdown (before the first ## heading)
    ## Section title {#id}
    markdown, raw HTML, ```python cells``` (attrs: norun="why" dl="what it downloads" error hide),
    :::note|tip|warn|exam|fix … :::     callouts
    :::deep Summary … :::                collapsible
    :::pq answer="A, C" type="single|multi|num|text" tag="…"   question @@ option @@ option @@why explanation :::
                                     (text: answer="Firm|Firms" lists the accepted words, compared ignoring case)
Subject index: <subject-src-dir>/_index.md (title, eyebrow, lead, badge, then markdown; cards are added).
"""
import ast, base64, contextlib, html, io, json, os, re, sys, traceback, warnings
from markdown_it import MarkdownIt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
SRC, PROG = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
ONLY = set(sys.argv[3:])
SUBJ_DIR = os.path.basename(SRC)                       # e.g. 00-sklearn-basics
OUT = os.path.abspath(os.environ.get("NOTES_OUT") or os.path.join(PROG, SUBJ_DIR))   # a subject may live outside programming/ (e.g. foundation/)
SUBJ_KEY = "dl" + SUBJ_DIR[:3]
ASSETS = os.path.relpath(os.path.join(PROG, "assets"), OUT).replace(os.sep, "/")      # "../assets" for programming/<subject>
HOME = os.path.relpath(os.path.join(PROG, "index.html"), OUT).replace(os.sep, "/")
REPORT = []

# ------------------------------------------------------------------ markdown with maths protected
md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table").enable("strikethrough")
def fence(self, tokens, idx, options, env):
    t = tokens[idx]
    lang = (t.info or "text").strip().split()[0].lower()
    label = re.search(r'label="([^"]*)"', t.info or "")
    lab = f' data-label="{html.escape(label.group(1))}"' if label else (' data-label="python"' if lang == "py" else "")
    return f'<div class="codewrap"{lab}><pre class="code" data-lang="{lang}">{html.escape(t.content.rstrip(chr(10)))}</pre></div>\n'
md.add_render_rule("fence", fence)
_tbl_open = md.renderer.rules.get("table_open")
md.add_render_rule("table_open", lambda self, t, i, o, e: '<div class="tblwrap"><table>\n')
md.add_render_rule("table_close", lambda self, t, i, o, e: '</table></div>\n')

MATH_D = re.compile(r"\$\$(.+?)\$\$", re.S)
MATH_I = re.compile(r"(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<![\s\\])\$(?![\w$])")
def protect(text):
    """Pull $…$ / $$…$$ out before markdown sees them (underscores and asterisks inside maths)."""
    store = []
    def keep(kind):
        def f(m):
            store.append((kind, m.group(1).strip()))
            return f"{len(store) - 1}"     # private-use characters: markdown-it keeps them (it replaces NUL)
        return f
    # leave code spans and fences alone
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]+`)", text)
    for i in range(0, len(parts), 2):
        parts[i] = MATH_I.sub(keep("i"), MATH_D.sub(keep("d"), parts[i]))
    return "".join(parts), store
def restore(s, store):
    def put(m):
        kind, tex = store[int(m.group(1))]
        return f'<div class="tex-d">{html.escape(tex)}</div>' if kind == "d" else f'<span class="tex">{html.escape(tex)}</span>'
    s = re.sub(r"<p>(\d+)</p>", put, s)
    return re.sub(r"(\d+)", put, s)
def render(text, inline=False):
    t, store = protect(text)
    out = md.renderInline(t) if inline else md.render(t)
    return restore(out, store)

def slug(s, used):
    s = re.sub(r"<[^>]+>", "", s); s = html.unescape(s).lower()
    s = re.sub(r"`", "", s); s = re.sub(r"[^\w\s-]", "", s, flags=re.U).strip()
    s = re.sub(r"[\s_]+", "-", s) or "section"
    base, n = s, 2
    while s in used: s, n = f"{base}-{n}", n + 1
    used.add(s); return s

# ------------------------------------------------------------------ running the cells (one namespace per page)
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.formatwarning = lambda m, c, f, l, line=None: f"{c.__name__}: {m}\n"   # no local file paths in outputs
# Never let a cell open the user's web browser: outside Jupyter, plotly's fig.show() and webbrowser.open()
# start a local server and open a browser tab. Both are turned into no-ops for the build.
import webbrowser
webbrowser.open = webbrowser.open_new = webbrowser.open_new_tab = lambda *a, **k: False
os.environ["PLOTLY_RENDERER"] = "json"
try:
    import plotly.io as _pio, plotly.basedatatypes as _pbd
    _pio.show = lambda *a, **k: None
    _pbd.BaseFigure.show = lambda self, *a, **k: None
except ImportError:
    pass
_URLS = {u:os.path.join(PROG, "assets", "data", f) for u, f in json.load(open(os.path.join(PROG, "assets", "data", "urls.json"))).items()}
_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localdata.json")
if os.path.exists(_LOCAL):      # datasets kept out of git (licence): read from tools/notes/data
    _URLS.update({k: os.path.join(PROG, "..", v) for k, v in json.load(open(_LOCAL)).items()})
_read_csv = pd.read_csv
def _local_read_csv(path, *a, **k):
    if isinstance(path, str) and path in _URLS and os.path.exists(_URLS[path]): path = _URLS[path]
    return _read_csv(path, *a, **k)
pd.read_csv = _local_read_csv
STORED = {}   # stem -> {cell no: rows} for pages built from the notebook's own outputs

class Kernel:
    def __init__(self, page, imgdir, rel):
        self.ns = {"__name__": "__main__"}
        self.page, self.imgdir, self.rel, self.nimg = page, imgdir, rel, 0
        plt.close("all"); plt.rcParams.update(plt.rcParamsDefault); matplotlib.use("agg")
    def figs(self, out):
        for n in plt.get_fignums():
            self.nimg += 1
            name = f"{self.page}-{self.nimg:02d}.png"
            os.makedirs(self.imgdir, exist_ok=True)
            plt.figure(n).savefig(os.path.join(self.imgdir, name), format="png", bbox_inches="tight", dpi=96)
            out.append(["png", f"{self.rel}/{name}"])
        plt.close("all")
    def run(self, src):
        out = []
        class W(io.TextIOBase):
            def __init__(s, kind): s.kind = kind
            def write(s, x):
                if not x: return 0
                if out and out[-1][0] == s.kind: out[-1][1] += x
                else: out.append([s.kind, x])
                return len(x)
        orig_show = plt.show
        plt.show = lambda *a, **k: self.figs(out)
        try:
            with contextlib.redirect_stdout(W("text")), contextlib.redirect_stderr(W("err")):
                tree = ast.parse(src, "<cell>", "exec")
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<cell>", "exec"), self.ns)
                if last is not None:
                    v = eval(compile(ast.Expression(last.value), "<cell>", "eval"), self.ns)
                    if v is not None and not src.rstrip().endswith(";"):
                        # tables as HTML (pandas); everything else, estimators included, as its repr
                        h = getattr(v, "_repr_html_", None) if type(v).__module__.startswith("pandas") else None
                        s = h() if callable(h) else None
                        out.append(["html", s] if s else ["text", repr(v) + "\n"])
            self.figs(out)
            ok = True
        except BaseException as e:
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            keep = [l for l in tb if 'File "<cell>"' in l or not l.startswith("  File")]
            keep = [l for l in keep if not l.startswith("Traceback") or True]
            out.append(["err", "".join(tb[:1] + [l for l in tb[1:] if '"<cell>"' in l] + tb[-1:])])
            self.figs(out)
            ok = False
        finally:
            plt.show = orig_show
        # drop matplotlib's noisy return values (<matplotlib.…>, Text(…), [<…>]) when a plot was drawn
        if any(k == "png" for k, _ in out):
            out = [o for o in out if not (o[0] == "text" and re.match(r"^(\[?<matplotlib|Text\(|\(array\(|<seaborn)", o[1]))]
        return out, ok

def clean_html_table(s):
    s = re.sub(r"<style[\s\S]*?</style>", "", s)
    s = re.sub(r'\s(border|class)="[^"]*"', "", s)
    s = re.sub(r"<table[^>]*>", "<table>", s)
    s = s.replace("<div>", "").replace("</div>", "")
    return s.strip()

def out_html(rows):
    parts = []
    for kind, val in rows:
        if kind == "png": parts.append(f'<div class="orow plot"><img src="{html.escape(val)}" alt="plot" loading="lazy"></div>')
        elif kind == "html": parts.append(f'<div class="orow df">{clean_html_table(val)}</div>')
        else:
            v = val.rstrip("\n")
            if v: parts.append(f'<div class="orow"><pre{" class=err" if kind == "err" else ""}>{html.escape(v)}</pre></div>')
    return "".join(parts)

# ------------------------------------------------------------------ blocks
ATTR = re.compile(r'(\w+)(?:="([^"]*)")?')
def attrs(s): return {k: (v if v is not None else True) for k, v in ATTR.findall(s or "")}

ICONS = {"note": "ℹ️", "tip": "💡", "warn": "⚠️", "exam": "🎯", "fix": "🛠️"}
def pq_html(a, body):
    chunks = re.split(r"^@@(why)?\s*$", body, flags=re.M)
    q = chunks[0]; opts, why = [], ""
    i = 1
    while i < len(chunks):
        tag, text = chunks[i], chunks[i + 1]
        if tag == "why": why = text
        else: opts.append(text)
        i += 2
    lis = []
    for o in opts:
        o = o.strip("\n")
        note = ""
        m = re.search(r"^~\s?(.*)$", o, re.M | re.S)
        if m: note, o = m.group(1), o[:m.start()].rstrip()
        inner = render(o).replace('<div class="codewrap"', '<div class="codewrap" data-nohead')   # no header bar inside an answer option
        inner = re.sub(r"^<p>([\s\S]*)</p>\s*$", r"\1", inner.strip()) if inner.count("<p>") == 1 else inner
        lis.append(f'<li>{inner}{f"<span class=pq-note>{render(note, True)}</span>" if note else ""}</li>')
    t = a.get("type", "multi" if "," in a.get("answer", "") else "single")
    extra = "".join(f' data-{k}="{html.escape(str(v))}"' for k, v in a.items() if k in ("tol", "tag"))
    return (f'<div class="pq" data-answer="{html.escape(a["answer"])}" data-type="{t}"{extra}>'
            f'<div class="pq-q">{render(q.strip())}</div>' + (f'<ol class="pq-opts">{"".join(lis)}</ol>' if lis else "") +
            f'<div class="pq-why">{render(why.strip())}</div></div>\n')

def body_html(text, kernel, page):
    """Markdown with python cells (run), callouts, questions and collapsibles."""
    out, buf = [], []
    def flush():
        if buf: out.append(render("".join(buf))); buf.clear()
    lines = text.splitlines(keepends=True); i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^```python\b(.*)$", ln)
        if m:
            a = attrs(m.group(1)); j = i + 1; code = []
            while not lines[j].startswith("```"): code.append(lines[j]); j += 1
            src = "".join(code).rstrip("\n")
            flush(); i = j + 1
            if a.get("rec"): rows, ok = STORED.get(page, {}).get(a["rec"], []), True
            else: rows, ok = kernel.run(src) if not a.get("skip") else ([], True)
            if not ok and not a.get("error"): REPORT.append(f"{page}: cell failed:\n{src[:200]}\n{rows[-1][1][-400:]}")
            if a.get("hide"): continue
            if a.get("fig"):          # a diagram: the picture, with the (runnable) code that draws it folded away
                pngs = [v for kd, v in rows if kd == "png"]
                cap = a["fig"] if isinstance(a["fig"], str) else ""
                out.append("".join(f'<figure class="fig"><img src="{html.escape(v)}" alt="{html.escape(cap)}" loading="lazy">'
                                   + (f"<figcaption>{render(cap, True)}</figcaption>" if cap else "") + "</figure>\n" for v in pngs)
                           + f'<details class="figcode"><summary>Python code for this figure</summary><div class="codewrap pyc"><pre class="code" data-lang="python">{html.escape(src)}</pre></div></details>\n')
                continue
            extra ="".join(f' data-{k}="{html.escape(str(v))}"' for k, v in a.items() if k in ("norun", "dl"))
            rec = out_html(rows)
            out.append(f'<div class="codewrap pyc"{extra}><pre class="code" data-lang="python">{html.escape(src)}</pre>'
                       + (f'<div class="outblk rec"><div class="oh">Recorded output</div>{rec}</div>' if rec else "") + "</div>\n")
            continue
        m = re.match(r"^:::(\w+)\s*(.*)$", ln)
        if m:
            kind, rest = m.group(1), m.group(2); j = i + 1; inner = []; depth = 0
            while True:
                if re.match(r"^:::\w", lines[j]): depth += 1
                elif lines[j].strip() == ":::":
                    if depth == 0: break
                    depth -= 1
                inner.append(lines[j]); j += 1
            flush(); i = j + 1
            content = "".join(inner)
            if kind == "pq": out.append(pq_html(attrs(rest), content))
            elif kind == "deep": out.append(f'<details class="deep"><summary>{render(rest, True)}</summary><div class="body">{body_html(content, kernel, page)}</div></details>\n')
            else: out.append(f'<div class="callout {kind}"><div class="ic">{ICONS.get(kind, "ℹ️")}</div><div>{body_html(content, kernel, page)}</div></div>\n')
            continue
        buf.append(ln); i += 1
    flush()
    return "".join(out)

def front(text):
    m = re.match(r"^---\n([\s\S]*?)\n---\n", text)
    meta = {}
    if m:
        for l in m.group(1).splitlines():
            if ":" in l: k, v = l.split(":", 1); meta[k.strip()] = v.strip()
        text = text[m.end():]
    return meta, text

PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · {subject} · ML Notes</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="{assets}/notes.css">
</head>
<body data-subject="{skey}" data-page="{stem}"{data}>
<div class="progress" id="progress"></div>
<header class="topbar">
  <button class="iconbtn" id="menuBtn" aria-label="Contents">☰</button>
  <div class="crumbs"><a href="{home}">ML</a> · <a href="index.html">{subject}</a> · <b>{title}</b></div>
  <div class="spacer"></div>
  <span class="pill" id="readTime"></span>
  <button class="iconbtn" id="themeBtn" title="Toggle light / dark">🌗</button>
</header>
<div class="layout">
<aside class="toc" id="toc">
  <h4>On this page</h4>
  <nav id="tocNav"></nav>
  <div class="meter">
    <div><b id="doneCount">0</b> / <span id="doneTotal">0</span> sections understood</div>
    <div class="bar"><i id="doneBar"></i></div>
    <div style="margin-top:10px" hidden>❓ Practice: <b id="pqTally">0 / 0</b> correct</div>
  </div>
</aside>
<main><div class="content">
<div class="hero"><div class="js-badge">{badge}</div><div class="eyebrow">{eyebrow}</div><h1>{title_h}</h1><div class="lead">{lead}</div></div>
{body}
{pager}
</div></main>
</div>
<script src="{assets}/notes.js"></script>
</body>
</html>
"""

def build_page(path, subject_title, nav):
    stem = os.path.splitext(os.path.basename(path))[0]
    meta, text = front(open(path, encoding="utf-8").read())
    k = Kernel(stem, os.path.join(OUT, "images", "out"), "images/out")
    side = os.path.join(SRC, stem + ".outputs.json")
    if os.path.exists(side): STORED[stem] = json.load(open(side, encoding="utf-8"))
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    used, secs = set(), []
    body = ""
    pre = parts[0].strip()
    if pre: body += f'<div class="preface">{body_html(pre, k, stem)}</div>\n'
    for n, i in enumerate(range(1, len(parts), 2), 1):
        head = parts[i].strip(); content = parts[i + 1]
        m = re.search(r"\s*\{#([\w-]+)\}\s*$", head)
        sid = m.group(1) if m else None
        if m: head = head[:m.start()]
        sid = sid or slug(head, used); used.add(sid)
        title_plain = html.unescape(re.sub(r"<[^>]+>", "", render(head, True)))
        num = f"{n:02d}"
        inner = body_html(content, k, stem)
        # ### headings get ids for the contents
        hu = set()
        inner = re.sub(r"<h3>(.*?)</h3>", lambda mm: f'<h3 id="{sid}-{slug(mm.group(1), hu)}">{mm.group(1)}</h3>', inner)
        body += (f'<section class="sec" id="{sid}" data-num="{num}" data-title="{html.escape(title_plain)}">\n'
                 f'<h2><span class="num">{num}</span> {render(head, True)}</h2>\n{inner}'
                 f'<div class="donebar"><button class="donebtn" data-done="{sid}">Mark as understood</button></div>\n</section>\n\n')
        secs.append(sid)
    i = nav.index(stem)
    prev = f'<a href="{nav[i-1]}.html"><small>← Previous</small>{html.escape(TITLES[nav[i-1]])}</a>' if i else f'<a href="index.html"><small>← Back to</small>{html.escape(subject_title)} index</a>'
    nxt = f'<a class="next" href="{nav[i+1]}.html"><small>Next →</small>{html.escape(TITLES[nav[i+1]])}</a>' if i + 1 < len(nav) else f'<a class="next" href="index.html"><small>Back to →</small>{html.escape(subject_title)} index</a>'
    data = f' data-py-data="{meta["data"]}"' if meta.get("data") else ""
    page = PAGE.format(title=html.escape(meta["title"]), title_h=render(meta["title"], True), subject=html.escape(subject_title), skey=SUBJ_KEY, stem=stem, data=data, assets=ASSETS, home=HOME,
                       badge=html.escape(meta.get("badge", "ML")), eyebrow=html.escape(meta.get("eyebrow", "")), lead=render(meta.get("lead", "")),
                       body=body, pager=f'<div class="pager">{prev}{nxt}</div>')
    open(os.path.join(OUT, stem + ".html"), "w", encoding="utf-8", newline="\n").write(page)
    return {"stem": stem, "title": meta["title"], "blurb": meta.get("blurb", ""), "kind": meta.get("kind", "note"), "group": meta.get("group", ""), "sections": len(secs), "cells": k.nimg}

TITLES = {}
def main():
    files = sorted(f for f in os.listdir(SRC) if re.match(r"\d\d_.*\.md$", f))
    nav = [os.path.splitext(f)[0] for f in files]
    for f in files: TITLES[os.path.splitext(f)[0]] = front(open(os.path.join(SRC, f), encoding="utf-8").read())[0]["title"]
    imeta, itext = front(open(os.path.join(SRC, "_index.md"), encoding="utf-8").read())
    subject_title = imeta["title"]
    built = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if ONLY and stem not in ONLY: fm = front(open(os.path.join(SRC, f), encoding="utf-8").read())[0]; built.append({"stem": stem, "title": TITLES[stem], "blurb": fm.get("blurb", ""), "kind": fm.get("kind", "note"), "group": fm.get("group", ""), "sections": None}); continue
        print("building", stem)
        built.append(build_page(os.path.join(SRC, f), subject_title, nav))
    # subject index
    card = lambda bs: "".join(f'<a class="ncard" href="{b["stem"]}.html" data-progress="{SUBJ_KEY}/{b["stem"]}"><div class="no">{b["stem"][:2]}</div><h5>{html.escape(b["title"])}</h5>'
                    f'<p>{render(b["blurb"], True)}</p><div class="pb"><i></i></div><div class="pt">not started</div></a>' for b in bs)
    k = Kernel("index", os.path.join(OUT, "images", "out"), "images/out")
    itext = itext.replace("{{cards}}", f'<div class="cards">{card(built)}</div>')
    # {{cards:note}} / {{cards:lab}} pick pages by kind; any other {{cards:x}} picks pages whose front matter has group: x
    itext = re.sub(r"\{\{cards:([\w-]+)\}\}", lambda m: f'<div class="cards">{card([b for b in built if (b.get("kind", "note") if m.group(1) in ("note", "lab") else b.get("group")) == m.group(1)])}</div>', itext)
    ibody = body_html(itext, k, "index")
    page = PAGE.replace('<aside class="toc" id="toc">', '<aside class="toc" id="toc" style="display:none">').format(
        title=html.escape(subject_title), title_h=html.escape(subject_title), subject=html.escape(subject_title), skey=SUBJ_KEY, stem="index", data="", assets=ASSETS, home=HOME,
        badge=html.escape(imeta.get("badge", "ML")), eyebrow=html.escape(imeta.get("eyebrow", "")), lead=render(imeta.get("lead", "")), body=ibody,
        pager=f'<div class="pager"><a href="{HOME}"><small>← Back to</small>All ML subjects</a></div>')
    page = page.replace('<div class="layout">', '<div class="layout" style="grid-template-columns:minmax(0,1fr)">')
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8", newline="\n").write(page)
    print("\n".join(REPORT) if REPORT else "all cells ran as expected")

if __name__ == "__main__":
    main()
