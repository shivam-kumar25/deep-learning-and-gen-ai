"""Self-test for worker.py: run a few cells through the real protocol and print the events.
Usage: .venv/Scripts/python tools/kernel/selftest.py"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
cells = [
    "import torch\nx = torch.randn(3, device='cuda' if torch.cuda.is_available() else 'cpu')\nprint(x.device)\nfor i in range(3): print('epoch', i)\nround(float(x.sum()), 3) * 0 + 7",
    "1/0",
    "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3]); plt.show()",
    "y = 41\n",
    "y + 1",
]
p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
print(p.stdout.readline().strip())
for i, c in enumerate(cells, 1):
    p.stdin.write(json.dumps({"id": i, "code": c}) + "\n"); p.stdin.flush()
    while True:
        ev = json.loads(p.stdout.readline())
        shown = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v) for k, v in ev.items()}
        print(shown)
        if ev.get("done"):
            break
p.stdin.close(); p.wait(timeout=30)
print("stderr tail:", p.stderr.read()[-300:])
