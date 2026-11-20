"""Convert a subject's notebooks into authored pages (markdown + python cells) for build.py.

Usage: python nb2md.py <spec.json>
spec: {"subject": "<programming>/<NN-subject>", "out": "<src dir>",
       "pages": [{"nb": "rel/path.ipynb", "stem": "NN_slug", "kind": "note|lab", "run": true|false,
                  "title": opt, "lead": opt, "blurb": opt, "badge": opt, "data": opt, "drop_sections": [regex...]}]}
run=false keeps the outputs stored in the notebook (written to <stem>.outputs.json); run=true lets build.py run the cells.
Prints a review report: empty sections, error outputs, leftover .ipynb links, <details> blocks, magics.
"""
import base64, json, os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
spec = json.load(open(sys.argv[1], encoding="utf-8"))
SUBJ, OUT = spec["subject"], spec["out"]
os.makedirs(OUT, exist_ok=True)
REPORT = []

EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u20E3]+")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PATHS = re.compile(r"(?:[A-Za-z]:\\Users\\[^\s'\"]+|/usr/(?:local/)?lib/python3[^\s'\"]*|/opt/conda/[^\s'\"]*|/home/[^\s'\"]+|/Users/[^\s'\"]+|C:\\Python\d+\\[^\s'\"]*|/tmp/ipykernel_[^\s'\"]*)")
def sanitize(s):
    s = ANSI.sub("", s)
    s = PATHS.sub(lambda m: "…" + re.split(r"[\\/]", m.group(0))[-1], s)
    return s

def clean_heading(t):
    t = t.strip()
    t = re.sub(r"^\*\*STEP\s*#?\s*(\d+)\*\*\s*[.:]?\s*:?\s*", r"Step \1 · ", t, flags=re.I)     # "**STEP #2**. Data exploration"
    t = re.sub(r"^\*\*(.*)\*\*$", r"\1", t).strip()
    if t.count("`") % 2: t = t.replace("`", "")          # an unbalanced backtick would leak into the page
    t = re.sub(r"^\*+|\*+$", "", t).strip()
    t = re.sub(r"^`([^`]+)`$", r"`\1`", t)
    t = EMOJI.sub("", t).strip()
    t = re.sub(r"^(?:\d+(?:\.\d+)*\s*[·.:)\-]\s*)", "", t).strip()      # "1 · ", "2.3 · ", "1. "
    t = re.sub(r"^\*\*STEP\s*#?\s*(\d+)\*\*\.?\s*", r"Step \1 · ", t, flags=re.I)
    t = re.sub(r"\s*<br\s*/?>\s*$", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" -·")
    return t or "Section"

def fix_img(path, nbdir):
    if re.match(r"^(https?:|data:)", path): return path
    from urllib.parse import unquote, quote
    target = os.path.normpath(os.path.join(SUBJ, nbdir, unquote(path)))
    if not os.path.exists(target):      # a stale path: look for the file by name inside the subject
        name = os.path.basename(unquote(path))
        hits = [os.path.join(dp, name) for dp, dn, fn in os.walk(SUBJ) if name in fn] or \
               [os.path.join(dp, name) for dp, dn, fn in os.walk(os.path.dirname(SUBJ)) if name in fn]     # or in another subject
        if hits: target = hits[0]
    rel = os.path.relpath(target, SUBJ).replace("\\", "/")
    if not os.path.exists(target): REPORT.append(f"   missing image {path}")
    return quote(rel)

class _M:      # a tiny stand-in for a regex match: group(1) = alt, group(2) = path
    def __init__(self, alt, path): self.g = (None, alt, path)
    def group(self, i): return self.g[i]
def md_images(s, fn):
    """![alt](path "title") with any bracket nesting in alt"""
    out, i = [], 0
    while True:
        j = s.find("![", i)
        if j < 0: out.append(s[i:]); break
        depth, k = 0, j + 1
        while k < len(s):
            if s[k] == "[": depth += 1
            elif s[k] == "]":
                depth -= 1
                if depth == 0: break
            k += 1
        m = re.match(r"\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", s[k + 1:]) if k < len(s) else None
        if not m: out.append(s[i:j + 2]); i = j + 2; continue
        out.append(s[i:j]); out.append(fn(_M(s[j + 2:k], m.group(1)))); i = k + 1 + m.end()
    return "".join(out)

ADMON = {"note": "note", "info": "note", "question": "exam", "tip": "tip", "important": "warn", "warning": "warn", "danger": "warn", "example": "note", "success": "tip",
         "quote": "note", "abstract": "note", "summary": "note", "hint": "tip", "caution": "warn", "attention": "warn", "bug": "warn", "failure": "warn"}
ADMON_RE = r"(?m)^!!!\s*(\w+)(?:\s+\"([^\"]*)\")?[ \t]*\n((?:(?:[ \t]+[^\n]*)?\n)*?)(?=^\S|\Z)"
def admonitions(s):
    """MkDocs boxes: '!!! note "Title"' (or '!!!note') followed by indented lines  →  ':::note' callouts.
    A box nested inside a box becomes a bold-headed paragraph of the outer one."""
    import textwrap
    def body_of(m): return textwrap.dedent(m.group(3).expandtabs(4)).strip()
    def inner(m): return "\n" + (f"**{m.group(2)}.** " if m.group(2) else "") + body_of(m) + "\n\n"
    def box(m):
        kind = m.group(1).lower()
        title = m.group(2) or ("Question" if kind == "question" else "")
        body = re.sub(ADMON_RE, inner, body_of(m) + "\n").strip()
        head = f"**{title}.** " if title else ""
        return f"\n:::{ADMON.get(kind, 'note')}\n{head}{body}\n:::\n"
    return re.sub(ADMON_RE, box, s + "\n")

def md_cell(src, nbdir):
    s = admonitions(src.replace("\r\n", "\n"))
    # decorative rules ("---" lines, used in stacks), but not table rules
    s = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", s)
    # images -> figures
    def img_md(m): return f'\n<figure><img src="{fix_img(m.group(2), nbdir)}" alt="{m.group(1).replace(chr(34), "&quot;")}" loading="lazy"></figure>\n'
    s = md_images(s, img_md)      # alt text may contain [nested [brackets]]
    def img_tag(m):
        src_ = re.search(r'src=["\']([^"\']+)', m.group(0)).group(1)
        alt = re.search(r'alt=["\']([^"\']*)', m.group(0))
        return f'\n<figure><img src="{fix_img(src_, nbdir)}" alt="{alt.group(1) if alt else ""}" loading="lazy"></figure>\n'
    s = re.sub(r"<img\b[^>]*>", img_tag, s)
    s = re.sub(r"(?m)^\{(?:width=)?\"?\d+(?:px|%)?\"?\}[ \t]*$", "", s)     # pandoc size attributes left after figures
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"(?m)^(\s*)```\s*python\b", r"\1```py", s)     # snippets in prose are shown, not run (only code cells run)
    return s.strip("\n")

MAGIC = re.compile(r"^\s*(%|!|\?|\w[\w.]*\?\s*$)")
def code_cell(src):
    lines = []
    for l in src.replace("\r\n", "\n").split("\n"):
        if re.match(r"^\s*%(matplotlib|config|load_ext|reload_ext|autoreload|time|timeit|%time|%%time)", l): continue
        if re.match(r"^\s*%%", l): continue
        m = re.match(r"^\s*\?\??\s*([\w.]+)\s*$", l) or re.match(r"^\s*([\w.]+)\s*\?\??\s*$", l)
        if m: lines.append(f"# In Jupyter: ?{m.group(1)}  (or help({m.group(1)}))"); continue
        if re.match(r"^\s*!", l): lines.append("# shell: " + l.strip()[1:]); continue
        lines.append(l)
    import textwrap
    return textwrap.dedent("\n".join(lines)).strip("\n")      # stray leading spaces gave IndentationErrors

def outputs(cell, stem, k, imgdir):
    rows = []
    for o in cell.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            txt = sanitize("".join(o.get("text", [])))
            if txt.strip(): rows.append(["err" if o.get("name") == "stderr" else "text", txt])
        elif t in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                k[0] += 1; name = f"{stem}-{k[0]:02d}.png"
                os.makedirs(imgdir, exist_ok=True)
                open(os.path.join(imgdir, name), "wb").write(base64.b64decode("".join(d["image/png"]) if isinstance(d["image/png"], list) else d["image/png"]))
                rows.append(["png", f"images/out/{name}"])
            elif "text/html" in d and ("<table" in "".join(d["text/html"])):
                rows.append(["html", "".join(d["text/html"])])
            elif "text/plain" in d:
                txt = sanitize("".join(d["text/plain"]))
                if re.match(r"^<(matplotlib|Figure|seaborn)[^>]*>$|^Text\(|^\[<matplotlib", txt.strip()): continue
                if "sklearn" in txt and "estimator" in str(d.get("text/html", "")) and "rerun this cell" in str(d.get("text/html", "")): pass
                rows.append(["text", txt + "\n"])
        elif t == "error":
            rows.append(["err", sanitize("\n".join(o.get("traceback", []))[-1500:] or f"{o.get('ename')}: {o.get('evalue')}")])
    return rows

NORUN = [(r"fetch_openml|fetch_20newsgroups|fetch_olivetti|fetch_lfw|fetch_covtype|fetch_kddcup|fetch_rcv1|fetch_species", "Downloads a dataset from the internet, which a browser tab cannot do from inside Python. The output was recorded on a computer."),
         (r"\bxgboost\b|\btensorflow\b|\btorch\b|\byfinance\b|\bgoogle\.colab\b|\bkaggle\b|\bmplfinance\b|\bplotly\b|\bgensim\b|\bspacy\b|\bnltk\b|\bimblearn\b|\bhdbscan\b|\bpmdarima\b|\bcv2\b", "Uses a library that is not available in the browser's Python. The output was recorded on a computer."),
         (r"read_csv\(\s*['\"](?!http)|open\(\s*['\"]|imread\(|load_img|Image\.open|read_excel\(|\.csv['\"]", "Reads a file from disk. The output was recorded on a computer."),
         (r"read_csv\(\s*['\"]http|urlopen|requests\.get|wget|urllib", "Downloads a file from the internet. The output was recorded on a computer.")]

NBMAP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbmap.json"), encoding="utf-8"))
THIS = os.path.basename(SUBJ)
def page_href(target):
    s, f = target.split("/")
    return f if s == THIS else f"../{s}/{f}"
def page_name(target):
    s, f = target.split("/")
    if f == "index.html": return f"{s[:2]} · {s[3:].replace('-', ' ').title()}"
    return f"{s[:2]} · note {f[:2]} ({f[3:-5].replace('_', ' ')})" if s != THIS else f"note {f[:2]} ({f[3:-5].replace('_', ' ')})"
def lookup(name):
    name = name.strip().strip("`").lstrip("./").replace("\\", "/")
    for key in (name, name.split("/")[-1]):
        if key in NBMAP: return NBMAP[key]
    base = name.split("/")[-1].lower()
    for key, v in NBMAP.items():      # old numbered names: "representations.ipynb" was "2_representations.ipynb"
        k = key.split("/")[-1].lower()
        if re.sub(r"^\d+_", "", k) == base or k.endswith("_" + base): return v
    return None
def relink(body):
    """links and mentions of the old notebooks → the pages that replaced them"""
    def mdlink(m):
        t = lookup(m.group(2))
        if not t: return m.group(0)
        text = m.group(1) if ".ipynb" not in m.group(1) else page_name(t)
        return f"[{text}]({page_href(t)})"
    body = re.sub(r"\[([^\]]*)\]\(([^)\s]*?\.ipynb)(?:#[^)\s]*)?\)", mdlink, body)
    def bare(m):
        t = lookup(m.group(1))
        return f"[{page_name(t)}]({page_href(t)})" if t else m.group(0)
    body = re.sub(r"`([\w./\- ]+\.ipynb)(?:\s*§[^`]*)?`", bare, body)
    parts = re.split(r"(```[\s\S]*?```|\]\([^)]*\))", body)      # plain-word mentions, outside code and link targets
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<![\w/.`\[])([\w\-]+\.ipynb)\b", bare, parts[i])
    def plain(m):      # inside code blocks (ASCII diagrams, comments): the new note's name, no link
        t = lookup(m.group(1))
        return page_name(t) if t else m.group(0)
    for i in range(1, len(parts), 2):
        if parts[i].startswith("```"): parts[i] = re.sub(r"(?:[\w\-]+/)*([\w\-]+\.ipynb)", plain, parts[i])
    return "".join(parts)
NARRATION = [   # course-material narration → plain teaching (no "the professor" / "the slide")
    (r"\*?Opening slide\s*[—-]\s*", ""), (r"\bthe deck's\b", "this note's"), (r"\bThe deck's\b", "This note's"),
    (r"\bthe deck\b", "this note"), (r"\bThe deck\b", "This note"), (r"\bthe professor's\b", "the course's"),
    (r"\bThe professor\b", "The course"), (r"\bthe professor\b", "the course"), (r"\bthis slide\b", "this figure"),
    (r"\bthe slide\b", "the figure"), (r"\bThe slide\b", "The figure"), (r"\bslide deck\b", "course material"),
    (r"In this colab, we will\b", "Here we"), (r"In this colab, we\b", "Here we"), (r"\bIn this colab\b", "Here"),
    (r"\bin this colab\b", "here"), (r"\bthis colab\b", "this lab"), (r"\bIn the last colab\b", "Earlier"), (r"\bthe last colab\b", "an earlier lab"),
    # recurring typos in the course notebooks
    (r"\btechnqiues\b", "techniques"), (r"\bDecission\b", "Decision"), (r"\bdemostrate\b", "demonstrate"), (r"\bSilhoutte\b", "Silhouette"),
    (r"\bBasline\b", "Baseline"), (r"\bdimention\b", "dimension"), (r"\bClassifer\b", "Classifier"), (r"\bclassifer\b", "classifier"),
]
def denarrate(body):
    parts = re.split(r"(```[\s\S]*?```)", body)
    for i in range(0, len(parts), 2):
        for rx, rep in NARRATION: parts[i] = re.sub(rx, rep, parts[i])
    return "".join(parts)

def modernise(code):
    """APIs that were removed from NumPy 2 / scikit-learn ≥ 1.6 / Matplotlib ≥ 3.6 (the notebooks predate them)"""
    code = re.sub(r"\bnp\.infty\b", "np.inf", code)
    code = re.sub(r"\bnp\.float_\b", "np.float64", code)
    code = code.replace("plt.style.use('seaborn')", "plt.style.use('seaborn-v0_8')").replace('plt.style.use("seaborn")', 'plt.style.use("seaborn-v0_8")')
    code = code.replace("store_cv_values", "store_cv_results").replace(".cv_values_", ".cv_results_")
    # mean_squared_error(..., squared=False) → root_mean_squared_error(...)
    new = re.sub(r"\bmean_squared_error\(([^()]*(?:\([^()]*\)[^()]*)*?),\s*squared\s*=\s*False\s*\)", r"root_mean_squared_error(\1)", code)
    if new != code:
        code = new
        if "import root_mean_squared_error" not in code: code = "from sklearn.metrics import root_mean_squared_error\n" + code
    code = re.sub(r"(max_iter\s*=\s*)np\.ceil\(", r"\1int_ceil(", code)
    if "int_ceil(" in code and "def int_ceil" not in code:
        code = "import math\nint_ceil = lambda v: int(math.ceil(v))     # max_iter must be an int\n" + code
    code = code.replace("base_estimator=", "estimator=")      # BaggingClassifier / AdaBoost renamed it
    code = code.replace(".corr()", ".corr(numeric_only=True)")
    code = re.sub(r",\s*old_names\s*=\s*(False|True)", "", code)      # statsmodels removed AutoReg(old_names=…)
    code = re.sub(r"\btol\s*=\s*-\s*np\.inf\b", "tol=None", code)
    code = re.sub(r"delim_whitespace\s*=\s*True", r'sep=r"\\s+"', code)      # removed in pandas 3
    code = re.sub(r"""penalty\s*=\s*(['"])none\1""", "penalty=None", code)      # LogisticRegression: the string 'none' was removed
    # seaborn ≥ 0.12 takes x / y as keywords only
    code = re.sub(r"(sns\.(?:countplot|barplot|boxplot|violinplot|histplot|kdeplot|stripplot)\()\s*(['\"][^'\"]+['\"])\s*,", r"\1x=\2,", code)
    code = re.sub(r"(sns\.(?:scatterplot|lineplot|regplot|barplot)\()\s*([^,=()]+?)\s*,\s*([^,=()]+?)\s*,", r"\1x=\2, y=\3,", code)      # SGD*: tol=None (not -inf) turns the stopping rule off
    # pandas 2.2+ offset aliases: month / quarter / year END are ME / QE / YE ("M", "Q", "Y", "A" were removed)
    code = re.sub(r"""(resample\(\s*|freq\s*=\s*|asfreq\(\s*)(["'])(M|Q|Y|A)\2""", lambda m_: f"{m_.group(1)}{m_.group(2)}{ {'M': 'ME', 'Q': 'QE', 'Y': 'YE', 'A': 'YE'}[m_.group(3)]}{m_.group(2)}", code)      # pandas 2 no longer skips text columns silently
    # the plot_* helpers were removed in scikit-learn 1.2: Display.from_estimator replaces them
    for old, disp in [("plot_roc_curve", "RocCurveDisplay"), ("plot_confusion_matrix", "ConfusionMatrixDisplay"), ("plot_precision_recall_curve", "PrecisionRecallDisplay")]:
        code = re.sub(rf"(import[^\n]*)\b{old}\b", rf"\1{disp}", code)
        code = re.sub(rf"\b{old}\(", f"{disp}.from_estimator(", code)
        code = re.sub(rf"(?m)^(\s*from sklearn\.metrics import[^\n]*\b{disp}\b)(.*\b{disp}\b)", r"\1", code)
    code = re.sub(r"\bnormalize\s*=\s*(True|False)\s*,?\s*", "", code) if "LinearRegression(" in code else code
    return code

QUESTIONS = []
_urls = os.path.join(os.path.dirname(os.path.abspath(SUBJ)), "assets", "data", "urls.json")
if not os.path.exists(_urls):      # a subject outside programming/ (foundation/) shares programming's assets
    _urls = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "notes", "assets", "data", "urls.json")
SERVED = list(json.load(open(_urls)))
def to_pq(m, stem):
    """<details><summary>❓ Question … > **A.** …</summary> solution … ✅ Option C</details>  →  :::pq block"""
    summ, sol = m.group(1), m.group(2)
    if "Question" not in summ: return m.group(0)
    unhead = lambda t: re.sub(r"(?m)^#{1,6}\s+(.+)$", r"**\1**", t)     # headings inside a question would split the page
    summ, sol = unhead(summ), unhead(sol)
    summ = re.sub(r"\*\*`?❓\s*Question:?`?\*\*", "", summ)
    summ = re.sub(r"(?m)^\s*\*\*OPTIONS:?\*\*\s*$", "", summ)
    opts = list(re.finditer(r"(?m)^>?\s*\*\*\(?([A-H])[.)]?\*\*[.)]?\s*", summ))
    sol0 = re.sub(r"\*\*`?Solution:?`?\*\*", "", sol).strip()
    if not opts:     # an open question: show it as a worked example with its solution folded
        first = re.sub(r"\s+", " ", re.sub(r"```[\s\S]*?```", "", summ)).strip()[:90] or "Worked question"
        return "\n".join([f":::deep 🧩 Worked question: {first}", summ.strip().replace("```python", "```py"), "", "**Solution**", "", sol0.replace("```python", "```py"), ":::"])
    q = summ[:opts[0].start()].strip()
    options = []
    for i, o in enumerate(opts):     # an option runs to the next marker (it can be a multi-line code block)
        text = summ[o.end(): opts[i + 1].start() if i + 1 < len(opts) else len(summ)]
        text = "\n".join(re.sub(r"^>\s?", "", l) for l in text.strip("\n").split("\n")).strip()
        options.append(text)
    sol = re.sub(r"\*\*`?Solution:?`?\*\*", "", sol).strip()
    marks = set()
    for a in re.findall(r"Option\s*\**\s*([A-H])\b[^\n]{0,25}?✅", sol): marks.add(a)
    for a in re.findall(r"✅\s*\**\s*Option\s*\**\s*([A-H])\b", sol): marks.add(a)
    for a in re.findall(r"(?i)final answer[^\n]*?\b([A-H])\b", sol): marks.add(a)
    for row in re.findall(r"(?m)^\|\s*\**([A-H])[.)]?[^|\n]*\|[^\n]*\|\s*✅\s*\|?\s*$", sol): marks.add(row)
    ans = ", ".join(sorted(marks))
    multi = len(marks) > 1 or re.search(r"is\s*/\s*are|are correct|select all|which of the following options are", q, re.I)
    QUESTIONS.append({"page": stem, "q": q, "options": options, "answer": ans})
    fixed = spec.get("answers", {}).get(str(len(QUESTIONS)))
    if fixed:
        if ans and fixed != ans: REPORT.append(f"   !! answer corrected for Q{len(QUESTIONS)} in {stem}: {ans} → {fixed}")
        ans = marks = fixed; QUESTIONS[-1]["answer"] = fixed
        multi = multi or "," in fixed
    if not marks: REPORT.append(f"   !! no answer found for a question in {stem}: {q[:80]}")
    q = q.replace("```python", "```py"); options = [o.replace("```python", "```py") for o in options]; sol = sol.replace("```python", "```py")
    parts = [f':::pq answer="{ans or "?"}" type="{"multi" if multi else "single"}" tag="Practice"', q]
    for o in options: parts += ["@@", o]
    parts += ["@@why", sol, ":::"]
    return "\n".join(parts)

def convert(p):
    # a page is one notebook, or "parts": [{"nb": …, "cells": [first, last], "heading": opt}, …] glued together
    parts = p.get("parts") or [{"nb": p["nb"], "cells": None}]
    stem = p["stem"]; REPORT.append(f"== {stem}  ({', '.join(x['nb'] + (str(x['cells']) if x.get('cells') else '') for x in parts)})")
    cells = []
    for part in parts:
        nb = json.load(open(os.path.join(SUBJ, part["nb"]), encoding="utf-8"))
        cs = nb["cells"] if not part.get("cells") else nb["cells"][part["cells"][0]:part["cells"][1] + 1]
        for c in cs: c["_dir"] = os.path.dirname(part["nb"])
        if part.get("heading"): cells.append({"cell_type": "markdown", "source": [part["heading"]], "_dir": ""})
        cells += cs
    # heading levels in use
    heads = []
    for c in cells:
        if c["cell_type"] == "markdown":
            fenced = False
            for l in "".join(c["source"]).splitlines():
                if re.match(r"^\s*(```|~~~)", l): fenced = not fenced
                m = None if fenced else re.match(r"^(#{1,6})\s+\S", l)      # '# comments' in code blocks are not headings
                if m: heads.append(len(m.group(1)))
    title = p.get("title")
    first_h1 = True
    h1count = heads.count(1)
    base = 1 if h1count > 1 else 2          # level that becomes a page section
    if h1count == 1: base = 2
    if h1count == 0: base = min(heads) if heads else 2
    out, sidecar, k, n = [], {}, [0], 0
    imgdir = os.path.join(SUBJ, "images", "out")
    drop = [re.compile(r, re.I) for r in p.get("drop_sections", []) + [r"^(?:🗂️\s*)?contents$", r"table of contents", r"where this fits", r"references\s*&\s*where this fits", r"^outline$"]]
    dropping_level = None
    norun_from = None
    for c in cells:
        src = "".join(c["source"])
        if not src.strip(): continue
        if c["cell_type"] == "markdown":
            lines_out = []; fenced = False
            for l in md_cell(src, c["_dir"]).split("\n"):
                if re.match(r"^\s*(```|~~~)", l): fenced = not fenced
                m = None if fenced else re.match(r"^(#{1,6})\s+(.*)$", l)
                if m:
                    lvl, text = len(m.group(1)), clean_heading(m.group(2))
                    if lvl == 1 and first_h1 and h1count <= 1 and not out and not lines_out:     # only a leading H1 is the page title
                        first_h1 = False
                        title = title or text; continue
                    first_h1 = False
                    if dropping_level is not None:
                        if lvl <= dropping_level: dropping_level = None
                        else: continue
                    if any(d.search(text) for d in drop): dropping_level = lvl; continue
                    new = max(2, min(4, 2 + (lvl - base)))
                    lines_out.append("#" * new + " " + text)
                    continue
                if dropping_level is not None: continue
                lines_out.append(l)
            t = "\n".join(lines_out).strip()
            if t: out.append(t)
        else:
            if dropping_level is not None: continue
            code = code_cell(src)
            code = modernise(code)
            for old, new in p.get("replace", []): code = code.replace(old, new)      # fixes apply before the cell is classified
            if code.strip() in [d_.strip() for d_ in p.get("drop_cells", [])]: continue
            if any(s_ in code for s_ in p.get("drop_cells_containing", [])): continue      # scratch cells
            if not code.strip(): continue
            n += 1
            attrs = []
            probe = code
            for u in SERVED: probe = re.sub("['\"]" + re.escape(u) + "['\"]", "SERVED_URL", probe)     # datasets the notes keep a copy of
            for rx, why in NORUN:
                if re.search(rx, probe):
                    attrs.append(f'norun="{why}"')
                    if "not available" not in why or re.search(r"=\s*\w+\.\w*(load|read|fetch|download)", probe): norun_from = norun_from or n   # only data loading cascades
                    break
            else:
                if norun_from: attrs.append('norun="Needs data that a cell above loads outside the browser. The output was recorded on a computer."')
            keep = not p.get("run", True) or re.search(r"\b(gensim|spacy|hdbscan|pmdarima|tensorflow|surprise|wordcloud|mplfinance|google\.colab)\b|files\.upload|drive\.mount", code)
            if keep and p.get("run", True) and re.search(r"except\s+(ImportError|ModuleNotFoundError|Exception)", code): keep = False   # a guarded import runs either way
            if keep:      # the notebook's own output (whole page, or one cell that needs a library the builder doesn't have)
                rows = outputs(c, stem, k, imgdir)
                sidecar[str(n)] = rows
                attrs.append(f'rec="{n}"')
                if any(r[0] == "err" and ("Traceback" in r[1] or "Error" in r[1]) for r in rows): REPORT.append(f"   cell {n}: stored ERROR → {rows[-1][1].strip().splitlines()[-1][:120]}")
            out.append("```python " + " ".join(attrs) + "\n" + code + "\n```")
    title = title or stem
    if p.get("setup"):      # pages cut out of a bigger notebook lose its imports
        out.insert(0, "```python\n" + p["setup"] + "\n```")
    body = "\n\n".join(out)
    # hand-written replacements for half-finished sections: {"section": "heading regex", "file": "x.md", "mode": "replace|after|end"}
    for pt in p.get("patches", []):
        add = open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), pt["file"]), encoding="utf-8").read().strip()
        if pt.get("mode") == "end": body += "\n\n" + add; continue
        mm = re.search(r"(?m)^## " + pt["section"] + r".*$", body)
        if not mm: REPORT.append(f"   !! patch target not found: {pt['section']}"); continue
        nxt = re.search(r"(?m)^## ", body[mm.end():])
        end = mm.end() + nxt.start() if nxt else len(body)
        if pt.get("mode") == "before": body = body[:mm.start()] + add + "\n\n" + body[mm.start():]
        elif pt.get("mode", "replace") == "replace": body = body[:mm.start()] + add + "\n\n" + body[end:]
        else: body = body[:end] + "\n\n" + add + "\n\n" + body[end:]
    # figures: no <figure> inside <figure>; an italic line right under an image becomes its caption
    body = re.sub(r"<figure>\s*(<figure>[\s\S]*?</figure>)\s*</figure>", r"\1", body)
    body = re.sub(r'(<figure><img [^>]*>)</figure>\s*\n\s*\*(?!\*)([^*][\s\S]{0,600}?)\*\s*(?=\n\n|\n?$)',
                  lambda mm: f'{mm.group(1)}<figcaption>{re.sub(chr(10), " ", mm.group(2).strip())}</figcaption></figure>', body)
    body = relink(body)
    body = denarrate(body)
    for old, new in p.get("replace", []):     # targeted fixes to cells and text
        if old not in body and new and new in body: continue      # already applied inside a code cell
        if old not in body: REPORT.append(f"   !! replace target not found: {old[:60]}"); continue
        body = body.replace(old, new)
    body = re.sub(r"<details>\s*<summary>([\s\S]*?)</summary>([\s\S]*?)</details>", lambda m: to_pq(m, stem), body)
    out = [body]
    meta = {"title": title, "eyebrow": p.get("eyebrow", ""), "badge": p.get("badge", "ML"), "kind": p.get("kind", "note"),
            "lead": p.get("lead", ""), "blurb": p.get("blurb", ""), "data": p.get("data", ""), "run": "yes" if p.get("run", True) else "no"}
    body = "\n\n".join(out)
    # report
    secs = re.split(r"(?m)^## ", body)
    for s in secs[1:]:
        head, _, rest = s.partition("\n")
        if len(re.sub(r"(?m)^#+ .*$", "", rest).strip()) < 40: REPORT.append(f"   thin/empty section: {head[:70]}")
    for m in re.findall(r"[\w\-. ]+\.ipynb", body): REPORT.append(f"   mentions notebook: {m.strip()}")
    if "<details" in body: REPORT.append(f"   has <details> blocks: {body.count('<details')} (questions?)")
    if re.search(r"❓|Question:", body): REPORT.append("   has QUESTION text")
    fm = "---\n" + "\n".join(f"{k_}: {v}" for k_, v in meta.items() if v) + "\n---\n"
    open(os.path.join(OUT, stem + ".md"), "w", encoding="utf-8", newline="\n").write(fm + body + "\n")
    if sidecar: json.dump(sidecar, open(os.path.join(OUT, stem + ".outputs.json"), "w", encoding="utf-8"), ensure_ascii=False)
    REPORT.append(f"   → {n} code cells, {len(secs) - 1} sections, title: {title}")

for p in spec["pages"]: convert(p)
print("\n".join(REPORT))
json.dump(QUESTIONS, open(os.path.join(OUT, "_questions.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n{len(QUESTIONS)} questions → {os.path.join(OUT, '_questions.json')}")
