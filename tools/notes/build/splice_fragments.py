"""Splice fragment blocks into note sources.
A fragment file holds blocks that start with a line
    %%% <note path relative to src/foundation> | before | <exact anchor line>
    %%% <note path> | afterpq | <tag of a pq; insert after its closing :::>
    %%% <note path> | append |
and run until the next %%% line. Each anchor must match exactly once."""
import sys, pathlib, re

SRC = pathlib.Path(sys.argv[1])
blocks = []
for frag in sys.argv[2:]:
    cur = None
    for line in pathlib.Path(frag).read_text(encoding="utf-8").splitlines():
        if line.startswith("%%% "):
            path, mode, anchor = (p.strip() for p in line[4:].split("|", 2))
            cur = [path, mode, anchor, []]; blocks.append(cur)
        elif cur is not None:
            cur[3].append(line)

for path, mode, anchor, body in blocks:
    f = SRC / path
    lines = f.read_text(encoding="utf-8").split("\n")
    text = "\n".join(body).strip("\n")
    if mode == "before":
        idx = [i for i, l in enumerate(lines) if l == anchor]
        assert len(idx) == 1, (path, anchor, idx)
        lines[idx[0]:idx[0]] = text.split("\n") + [""]
    elif mode == "afterpq":
        idx = [i for i, l in enumerate(lines) if l.startswith(":::pq") and f'tag="{anchor}"' in l]
        assert len(idx) == 1, (path, anchor, idx)
        j = idx[0] + 1
        while lines[j] != ":::":
            j += 1
        lines[j + 1:j + 1] = [""] + text.split("\n")
    elif mode == "append":
        while lines and lines[-1] == "":
            lines.pop()
        lines += [""] + text.split("\n") + [""]
    else:
        raise SystemExit("bad mode " + mode)
    f.write_text("\n".join(lines), encoding="utf-8")
    print(f"{path}: {mode} {anchor[:50]!r}  (+{len(text.splitlines())} lines)")
