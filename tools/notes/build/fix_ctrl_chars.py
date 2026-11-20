"""Repair LaTeX commands whose backslash escape was turned into a control character (\r, \t, \f, \b, \v, \a) by a shell heredoc."""
import sys
FIX = {"\r" + "ightarrow": r"\rightarrow", "\t" + "ext": r"\text", "\t" + "heta": r"\theta", "\f" + "rac": r"\frac",
       "\b" + "egin": r"\begin", "\x0b" + "ec": r"\vec", "\n" + "eq": None}
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8", newline="").read(); t = s
    for bad, good in FIX.items():
        if good: t = t.replace(bad, good)
    bad_left = [c for c in "\t\f\b\x0b\x07" if c in t]
    if "\r" in t.replace("\r\n", ""): bad_left.append("\\r")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(p, "fixed" if t != s else "unchanged", "| remaining control chars:", bad_left)
