"""Tidy converted headings: drop leading section numbers ("### 2 Applications", "## 6 Manipulating"), blockquote/bold
wrappers ("### > **Mathematical Definition**"), and stray bold. Usage: tidy_headings.py <file.md> ..."""
import re, sys

for f in sys.argv[1:]:
    t = open(f, encoding="utf-8").read()
    fm, body = re.match(r"(---\n.*?\n---\n)(.*)", t, re.S).groups()
    out, fenced = [], False
    for line in body.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        m = None if fenced else re.match(r"^(#{2,4}) +(.*)$", line)
        if m:
            h, text = m.groups()
            text = re.sub(r"^>\s*", "", text)
            text = text.replace("**", "").strip()
            text = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text)
            line = f"{h} {text}"
        out.append(line)
    open(f, "w", encoding="utf-8", newline="\n").write(fm + "\n".join(out))
    print("tidied", f)
