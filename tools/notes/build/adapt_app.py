"""One-off: turn the Studyloom app copied from the BDM repo into the deep-learning course's app (port 4300)."""
import pathlib, re, sys
R = pathlib.Path(__file__).resolve().parents[3]

def edit(rel, pairs, regex=False):
    p = R / rel; t = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if regex:
            t2 = re.sub(a, b, t, count=1, flags=re.S)
            if t2 == t: sys.exit(f"{rel}: pattern not found: {a[:70]!r}")
            t = t2
        else:
            if a not in t: sys.exit(f"{rel}: not found: {a[:70]!r}")
            t = t.replace(a, b)
    p.write_text(t, encoding="utf-8", newline="\n"); print("edited", rel)

SUBJECTS = '''export const SUBJECTS = [
  // each subject is a folder under notes/; tracks group them in the library
  { id: "d1-neural-network-foundations", num: "D1", track: "theory", title: "Neural Network Foundations", hue: 262, icon: "BrainCircuit", blurb: "What deep learning is and why it works; the neuron from biology to the perceptron; multi-layer perceptrons, the forward pass and the universal approximation theorem; activation functions." },
  { id: "d2-training-neural-networks", num: "D2", track: "theory", title: "Training Neural Networks", hue: 12, icon: "TrendingDown", blurb: "Loss functions, gradient descent and its variants (momentum, Adam), backpropagation worked by hand, the XOR network end to end, and regularisation against overfitting." },
  { id: "d3-pytorch", num: "D3", track: "pytorch", title: "PyTorch", hue: 22, icon: "Flame", blurb: "Tensors, shapes and dtypes, autograd, NumPy interop and the GPU; the PyTorch workflow: data, model, loss, optimiser, training loop, evaluation, saving and loading." },
  { id: "d4-convolutional-networks", num: "D4", track: "architectures", title: "Convolutional Networks", hue: 188, icon: "ScanSearch", blurb: "Why images need CNNs, convolution, kernel size, stride and padding, pooling, batch norm and dropout, building and training a CNN, and AlexNet, VGG, GoogLeNet and ResNet." },
  { id: "d5-projects", num: "D5", track: "projects", title: "Projects", hue: 142, icon: "Rocket", blurb: "End-to-end projects in PyTorch: Fashion-MNIST with an MLP, customer churn on tabular data, transfer learning with a pretrained ResNet, and U-Net image segmentation." },
];
'''
edit("server/indexer.js", [(r"export const SUBJECTS = \[.*?\n\];\n", SUBJECTS.replace("\\", "\\\\"))], regex=True)
edit("server/indexer.js", [('track: row.track || "economics"', 'track: row.track || "theory"')])
edit("server/db.js", [("DEFAULT 'economics'", "DEFAULT 'theory'")])
edit("server/index.js", [("process.env.PORT || 4200", "process.env.PORT || 4300")])
edit("server/papers.js", [('course: p.course || "BDM"', 'course: p.course || "DL"'), ('course: "BDM", exam: "Practice"', 'course: "DL", exam: "Practice"')])
edit("tools/notes/build/build.py", [('SUBJ_KEY = "bdm" + SUBJ_DIR[:3]', 'SUBJ_KEY = "dl" + SUBJ_DIR[:3]')])
edit("tools/notes/build/make_papers.mjs", [(r"const PAPERS = \[.*?\n\];\n", "const PAPERS = [];   // filled in as question papers arrive (see the BDM repo for the format)\n")], regex=True)
edit("package.json", [
    ('"name": "studyloom-bdm"', '"name": "studyloom-dl"'),
    ('"description": "Studyloom for Business Data Management: notes on economics, business, accounting and data analysis, with progress, search and an exam hall."',
     '"description": "Studyloom for Deep Learning and Gen AI: notes whose PyTorch cells run on a live Python kernel, with progress, search and an exam hall."'),
])
edit("client/src/pages/Library.jsx", [
    ('(s.track || "economics") === t.id', '(s.track || "theory") === t.id'),
    (r'  \{ id: "economics".*?\n  \{ id: "data".*?\},\n', '''  { id: "theory", title: "How neural networks learn", blurb: "From a single neuron to backpropagation: the ideas every model in this course is built on. Start here." },
  { id: "pytorch", title: "PyTorch", blurb: "The framework: tensors, autograd and the training workflow, run live on this computer." },
  { id: "architectures", title: "Architectures", blurb: "Convolutional networks for images, from the convolution operation to ResNet." },
  { id: "projects", title: "Projects", blurb: "End-to-end deep learning on real data: images, tables, transfer learning and segmentation." },
'''),
    ("Economics first, then business, then data.", "Neurons first, then PyTorch, then architectures and projects."),
])
edit("client/index.html", [
    ('content="Business Data Management: economics, business, accounting and data analysis, with the practice questions of the course worked out. Track your progress, keep notes, never lose your place."',
     'content="Deep learning with PyTorch: from the perceptron to CNNs, transfer learning and U-Net, with every cell runnable on a live Python kernel. Track your progress, keep notes, never lose your place."'),
    ("<title>Studyloom: business data management, one calm section at a time</title>", "<title>Studyloom: deep learning, one calm section at a time</title>"),
])
icons = R / "client/src/icons.js"; t = icons.read_text(encoding="utf-8")
if "ScanSearch" not in t: t = t.replace("export { ", "export { ScanSearch, Flame, TrendingDown, ", 1)
icons.write_text(t, encoding="utf-8", newline="\n")
for f in ("tools/notes/test/docker-test.sh", "tools/notes/test/exam-shot.mjs", "tools/notes/test/app-e2e.mjs"):
    p = R / f; p.write_text(p.read_text(encoding="utf-8").replace("4200", "4300"), encoding="utf-8", newline="\n")
print("done")
