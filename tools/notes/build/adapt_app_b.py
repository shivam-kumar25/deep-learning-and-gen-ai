"""One-off, second half of adapt_app.py (library tracks, page meta, icons, test ports)."""
import pathlib, re, sys
R = pathlib.Path(__file__).resolve().parents[3]

p = R / "client/src/pages/Library.jsx"; t = p.read_text(encoding="utf-8")
t = t.replace('(s.track || "economics") === t.id', '(s.track || "theory") === t.id')
t, n = re.subn(r'  \{ id: "economics".*?\n  \{ id: "data".*?\},\n', '''  { id: "theory", title: "How neural networks learn", blurb: "From a single neuron to backpropagation: the ideas every model in this course is built on. Start here." },
  { id: "pytorch", title: "PyTorch", blurb: "The framework: tensors, autograd and the training workflow, run live on this computer." },
  { id: "architectures", title: "Architectures", blurb: "Convolutional networks for images, from the convolution operation to ResNet." },
  { id: "projects", title: "Projects", blurb: "End-to-end deep learning on real data: images, tables, transfer learning and segmentation." },
''', t, count=1, flags=re.S)
if not n: sys.exit("tracks not found")
t = t.replace("Economics first, then business, then data.", "Neurons first, then PyTorch, then architectures and projects.")
p.write_text(t, encoding="utf-8", newline="\n")

p = R / "client/index.html"; t = p.read_text(encoding="utf-8")
t = re.sub(r'<meta name="description" content="[^"]*"', '<meta name="description" content="Deep learning with PyTorch: from the perceptron to CNNs, transfer learning and U-Net, with every cell runnable on a live Python kernel. Track your progress, keep notes, never lose your place."', t)
t = re.sub(r"<title>.*?</title>", "<title>Studyloom: deep learning, one calm section at a time</title>", t)
p.write_text(t, encoding="utf-8", newline="\n")

p = R / "client/src/icons.js"; t = p.read_text(encoding="utf-8")
if "ScanSearch" not in t: t = t.replace("export { ", "export { ScanSearch, Flame, TrendingDown, ", 1)
p.write_text(t, encoding="utf-8", newline="\n")
for f in ("tools/notes/test/docker-test.sh", "tools/notes/test/exam-shot.mjs", "tools/notes/test/app-e2e.mjs"):
    p = R / f; p.write_text(p.read_text(encoding="utf-8").replace("4200", "4300"), encoding="utf-8", newline="\n")
print("done")
