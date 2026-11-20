"""Polish converted notebook pages (nb2md output) into note sources.

Usage: python polish.py <spec.json>
spec: {"conv": dir of nb2md output, "src": note source dir, "notes_dir": built subject dir (kept slides are copied to its images/slides),
       "img_root": folder the image paths are relative to, "eyebrow": "Foundations", "badge": "🧠",
       "pages": {"02_neuron_and_perceptron": {"number": "02"}}, "renumber": {"Notebook 2": "note 03", ...}}
Slide images follow tools/notes/specs/images.json: "keep" (copied, captioned), "drop" (removed), "redraw" (a marker
<!-- REDRAW name | caption --> left for a figure cell). Unknown images are kept and reported.
Front matter gets eyebrow, blurb (the section titles) and lead (the italic opening line).
"""
import json, os, re, shutil, sys
from urllib.parse import unquote

spec = json.load(open(sys.argv[1], encoding="utf-8"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DECIDE = json.load(open(os.path.join(ROOT, "tools/notes/specs/images.json"), encoding="utf-8"))
os.makedirs(spec["src"], exist_ok=True)
slides = os.path.join(spec["notes_dir"], "images", "slides")
report = []

FIG = re.compile(r'(?:<div align="center">\s*)?<figure><img src="(?P<src>[^"]+)" alt="(?P<alt>[^"]*)"[^>]*>(?:<figcaption>(?P<cap2>.*?)</figcaption>)?</figure>\s*(?:<br\s*/?>\s*)?(?:<em>(?P<cap>.*?)</em>)?\s*(?:</div>)?', re.S)


def figure(m, stem):
    src, alt, cap = unquote(m.group("src")), m.group("alt"), (m.group("cap") or m.group("cap2") or "").strip()
    name = os.path.basename(src)
    what = DECIDE.get(name)
    if what is None:
        report.append(f"{stem}: undecided image {name} (kept)"); what = "keep"
    if what == "drop":
        return ""
    if what == "redraw":
        return f"\n<!-- REDRAW {name} | {cap or alt} -->\n"
    os.makedirs(slides, exist_ok=True)
    srcpath = os.path.join(ROOT, spec["img_root"], src)
    if os.path.exists(srcpath):
        shutil.copyfile(srcpath, os.path.join(slides, name))
    else:
        report.append(f"{stem}: missing image {srcpath}")
    capt = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'\n<figure><img src="images/slides/{name}" alt="{alt}" loading="lazy">{capt}</figure>\n'


for stem, meta in spec["pages"].items():
    t = open(os.path.join(spec["conv"], stem + ".md"), encoding="utf-8").read()
    fm, body = re.match(r"---\n(.*?)\n---\n(.*)", t, re.S).groups()
    title = re.search(r"^title:\s*(.*)$", fm, re.M).group(1).strip()
    body = FIG.sub(lambda m: figure(m, stem), body)
    # this course's cells run on the live kernel, which has PyTorch: "not available in the browser" cells become runnable
    body = re.sub(r'```python norun="Uses a library that is not available in the browser\'s Python\.[^"]*"', "```python", body)
    body = re.sub(r'```python norun="Needs data that a cell above loads outside the browser\.[^"]*"', "```python", body)
    body = re.sub(r"```python +\n", "```python\n", body)
    # the italic opening line is the note's lead
    lead = ""
    m = re.match(r"\s*\*([^*\n][^\n]*?)\*\s*\n", body)
    if m:
        lead, body = m.group(1).strip(), body[m.end():]
    # cross-references: "notebook 2" → "note 03"; link texts naming old folders → "note NN"
    for old, new in spec.get("renumber", {}).items():
        body = re.sub(rf"\b{old}\b", new, body)
    body = re.sub(r"\bthis notebook\b", "this note", body); body = re.sub(r"\bThis notebook\b", "This note", body)
    lead = re.sub(r"\bThis notebook\b", "This note", re.sub(r"\bthis notebook\b", "this note", lead))
    body = re.sub(r"\[`\.\./[^`\]]*`\]\(((?:\.\./[\w-]+/)?(\d\d)_[\w-]+\.html(?:#[\w-]+)?)\)", lambda m: f"[note {m.group(2)}]({m.group(1)})", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    heads = [re.sub(r"\s*\{#.*\}$", "", h) for h in re.findall(r"^## (.+)$", body, re.M)]
    blurb = " · ".join(h.split(" — ")[0].strip() for h in heads if not h.lower().startswith(("runnable demo", "where this fits")))
    front = (f"---\ntitle: {title}\neyebrow: {spec['eyebrow']} · note {meta['number']}\nbadge: {spec['badge']}\nkind: note\n"
             f"blurb: {blurb}\nlead: {lead}\nrun: yes\n---\n")
    open(os.path.join(spec["src"], stem + ".md"), "w", encoding="utf-8", newline="\n").write(front + body)
    print(f"{stem}: {len(heads)} sections, {body.count('<!-- REDRAW')} figures to redraw, {body.count('images/slides/')} slides kept")
print("\n".join(report) or "no report")
