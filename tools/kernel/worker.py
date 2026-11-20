"""A notebook-style Python kernel for Studyloom, one process per (user, note).

The Node server starts this script and talks to it over stdin/stdout with one JSON object per line:
  in : {"id": 7, "code": "x = 1\nprint(x)"}
  out: {"id": 7, "t": "text", "d": "1\n"}       streamed while the cell runs (t: text | err | png | html)
       {"id": 7, "done": true, "ok": true}       when the cell has finished
Cells share one namespace, like a Jupyter notebook. print() output streams as it happens, so a training
loop's progress shows epoch by epoch. matplotlib figures come back as PNGs; pandas tables as HTML; the value
of a cell's last expression is shown the way Jupyter shows it. Stopping a cell = the server kills this process.
"""
import ast, base64, io, json, os, sys, traceback, warnings

# Imports resolve from the working folder (workspace/, where dlviz.py and data/ live) the way a notebook's do,
# not from this script's folder.
sys.path[0] = os.getcwd()

# The protocol gets a private copy of the real stdout; file descriptor 1 is then pointed at stderr, so nothing a cell
# or a C library (CUDA, DataLoader workers) writes to the console can ever corrupt the JSON stream.
PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.__stdout__ = io.TextIOWrapper(os.fdopen(1, "wb", closefd=False), encoding="utf-8", line_buffering=True)
_CUR = {"id": None}


def emit(obj):
    PROTO.write(json.dumps(obj) + "\n")
    PROTO.flush()


class Stream:
    """stdout/stderr for a running cell: forwards each write as an event (\r progress bars keep their last line)."""
    def __init__(self, kind):
        self.kind, self.buf = kind, ""

    def write(self, s):
        if s and _CUR["id"] is not None:
            self.buf += s
            if "\n" in s or "\r" in s or len(self.buf) > 4000:    # whole lines (and progress-bar redraws) go out at once
                self.flush()
        return len(s or "")

    def flush(self):
        if self.buf and _CUR["id"] is not None:
            emit({"id": _CUR["id"], "t": self.kind, "d": self.buf})
        self.buf = ""

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


os.environ.setdefault("MPLBACKEND", "agg")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.formatwarning = lambda m, c, f, l, line=None: f"{c.__name__}: {m}\n"


def send_figures():
    sys.stdout.flush(); sys.stderr.flush()
    if "matplotlib.pyplot" not in sys.modules:
        return
    import matplotlib.pyplot as plt
    for n in plt.get_fignums():
        b = io.BytesIO()
        plt.figure(n).savefig(b, format="png", bbox_inches="tight", dpi=96)
        emit({"id": _CUR["id"], "t": "png", "d": base64.b64encode(b.getvalue()).decode()})
    plt.close("all")


def patch_show():
    if "matplotlib.pyplot" in sys.modules:
        import matplotlib.pyplot as plt
        if not getattr(plt.show, "_sl", False):
            def show(*a, **k):
                send_figures()
            show._sl = True
            plt.show = show


def show_value(v):
    if v is None:
        return
    mod = type(v).__module__
    if mod.startswith("pandas"):
        h = getattr(v, "_repr_html_", None)
        if callable(h):
            try:
                s = h()
                if s:
                    emit({"id": _CUR["id"], "t": "html", "d": s})
                    return
            except Exception:
                pass
    emit({"id": _CUR["id"], "t": "text", "d": repr(v) + "\n"})


NS = {"__name__": "__main__"}


def run(code):
    tree = ast.parse(code, "<cell>", "exec")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<cell>", "exec"), NS)
    patch_show()
    if last is not None:
        v = eval(compile(ast.Expression(last.value), "<cell>", "eval"), NS)
        if not code.rstrip().endswith(";"):
            show_value(v)
    patch_show()
    send_figures()


def main():
    out, err = Stream("text"), Stream("err")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        _CUR["id"] = msg.get("id")
        sys.stdout, sys.stderr = out, err
        ok = True
        try:
            run(msg.get("code", ""))
        except BaseException as e:                  # KeyboardInterrupt / SystemExit in a cell must not end the kernel
            ok = False
            out.flush(); err.flush()
            lines = traceback.format_exception(type(e), e, e.__traceback__)
            keep = [l for l in lines if "worker.py" not in l]
            emit({"id": _CUR["id"], "t": "err", "d": "".join(keep[-8:])})
            try:
                send_figures()
            except Exception:
                pass
        finally:
            out.flush(); err.flush()
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        emit({"id": _CUR["id"], "done": True, "ok": ok})
        _CUR["id"] = None


if __name__ == "__main__":
    emit({"ready": True, "python": sys.version.split()[0]})
    main()
