// Reads the HTML notes in the repo into the catalogue tables (subjects, notes, sections, search).
// Order and groups come from each subject's index.html, exactly as the notes present them.
// Run: npm run index   (the server also runs it on start when the catalogue is empty)
import fs from "node:fs";
import path from "node:path";
import { parse } from "node-html-parser";
import { db, contentFile, tx } from "./db.js";

export const SUBJECTS = [
  // each subject is a folder under notes/; tracks group them in the library
  { id: "d1-neural-network-foundations", num: "D1", track: "theory", title: "Neural Network Foundations", hue: 262, icon: "BrainCircuit", blurb: "What deep learning is and why it works; the neuron from biology to the perceptron; multi-layer perceptrons, the forward pass and the universal approximation theorem; activation functions." },
  { id: "d2-training-neural-networks", num: "D2", track: "theory", title: "Training Neural Networks", hue: 12, icon: "TrendingDown", blurb: "Loss functions, gradient descent and its variants (momentum, Adam), backpropagation worked by hand, the XOR network end to end, and regularisation against overfitting." },
  { id: "d3-pytorch", num: "D3", track: "pytorch", title: "PyTorch", hue: 22, icon: "Flame", blurb: "Tensors, shapes and dtypes, autograd, NumPy interop and the GPU; the PyTorch workflow: data, model, loss, optimiser, training loop, evaluation, saving and loading." },
  { id: "d4-convolutional-networks", num: "D4", track: "architectures", title: "Convolutional Networks", hue: 188, icon: "ScanSearch", blurb: "Why images need CNNs, convolution, kernel size, stride and padding, pooling, batch norm and dropout, building and training a CNN, and AlexNet, VGG, GoogLeNet and ResNet." },
  { id: "d5-projects", num: "D5", track: "projects", title: "Projects", hue: 142, icon: "Rocket", blurb: "End-to-end projects in PyTorch: Fashion-MNIST with an MLP, customer churn on tabular data, transfer learning with a pretrained ResNet, and U-Net image segmentation." },
];

const read = p => fs.readFileSync(contentFile(p), "utf8");
const exists = p => { const f = contentFile(p); return !!f && fs.existsSync(f); };
const clean = s => (s || "").replace(/\s+/g, " ").trim();
const posix = p => p.split(path.sep).join("/");

// a question's label for the index: its words, without the code it quotes
function qLabel(q) {
  const el = q.querySelector(".pq-q"); if (!el) return "Question";
  const c = parse(el.innerHTML); c.querySelectorAll("pre, .codewrap, figure, svg").forEach(x => x.replaceWith(" "));
  return clean(c.text).replace(/[,:]\s*$/, "").slice(0, 140) || "Question";
}

function pageInfo(rel) {
  const root = parse(read(rel), { blockTextElements: { script: false, style: false, pre: true, textarea: false, noscript: false } });
  const hero = root.querySelector(".hero");
  const title = clean(hero?.querySelector("h1")?.text || root.querySelector("h1")?.text || root.querySelector("title")?.text || rel);
  const eyebrow = clean(hero?.querySelector(".eyebrow")?.text);
  const lead = clean(hero?.querySelector(".lead")?.text).slice(0, 400);
  const allQ = root.querySelectorAll(".pq");   // numbered down the page, the way the notes number them (Q1, Q2 ...)
  const sections = root.querySelectorAll("section.sec[id]").map((s, i) => {
    const h2 = s.querySelector("h2");
    const questions = s.querySelectorAll(".pq").map(q => ({ n: allQ.indexOf(q) + 1, text: qLabel(q) }));
    const t = clean(s.getAttribute("data-title") || h2?.text.replace(/^\s*\d+\s*/, ""));
    s.querySelectorAll("script, style, .donebar, textarea").forEach(x => x.remove());
    return { id: s.getAttribute("id"), title: t, position: i, questions, body: clean(s.text).slice(0, 20000) };
  });
  const words = sections.reduce((n, s) => n + s.body.split(" ").length, 0);
  const html = root.toString();
  return {
    title, eyebrow, lead, sections,
    minutes: Math.max(2, Math.round(words / 200)),
    widgets: (html.match(/data-widget="/g) || []).length + (html.match(/class="demo"/g) || []).length + (html.match(/class="frame-item"/g) || []).length,
    runnable: (html.match(/class="codewrap pyc"/g) || []).length + (html.match(/data-play/g) || []).length,
    questions: (html.match(/class="pq"/g) || []).length,
  };
}

export function buildCatalogue({ quiet = false } = {}) {
  const subjects = [], notes = [], seen = new Set();
  SUBJECTS.forEach((s, si) => {
    const dir = s.dir || s.id;                       // the subject's folder under notes/
    const idx = `${dir}/index.html`;
    if (!exists(idx)) return;
    const root = parse(read(idx));
    const { dir: _dir, ...row } = s;
    subjects.push({ ...row, num: row.num || row.id.slice(0, 2), track: row.track || "theory", blurb: s.blurb || clean(root.querySelector(".hero .lead")?.text).slice(0, 300), index_path: idx, position: si });
    let pos = 0;
    // ML index pages: cards grouped under ### headings ("Notes", "Labs"); MAD-style pages used <section class="sec">
    let group = "Notes";
    for (const el of root.querySelectorAll("h3, section.sec h2, a.ncard[href]")) {
      if (el.tagName !== "A") { group = clean(el.text.replace(/^\s*\d+\s*/, "")) || group; continue; }
      const href = el.getAttribute("href");
      if (/^[a-z]+:/i.test(href)) continue;
      const rel = posix(path.normalize(path.join(dir, href.split("#")[0])));
      if (!rel.endsWith(".html") || seen.has(rel) || !exists(rel)) continue;
      seen.add(rel);
      const info = pageInfo(rel);
      const kind = /lab|worked example/i.test(group) || /_lab_/.test(rel) ? "lab" : "note";
      notes.push({ id: rel.replace(/\.html$/, ""), subject_id: s.id, path: rel, group_title: group, position: pos++, kind, ...info });
    }
  });
  tx(() => {
    db.exec("DELETE FROM search; DELETE FROM sections; DELETE FROM notes; DELETE FROM subjects;");
    const S = db.prepare("INSERT INTO subjects (id, num, title, blurb, hue, icon, index_path, position, track) VALUES (@id, @num, @title, @blurb, @hue, @icon, @index_path, @position, @track)");
    const N = db.prepare(`INSERT INTO notes (id, subject_id, path, title, eyebrow, lead, kind, group_title, position, minutes, widgets, runnable, questions)
                          VALUES (@id, @subject_id, @path, @title, @eyebrow, @lead, @kind, @group_title, @position, @minutes, @widgets, @runnable, @questions)`);
    const C = db.prepare("INSERT INTO sections (note_id, id, position, title, questions) VALUES (?, ?, ?, ?, ?)");
    const F = db.prepare("INSERT INTO search (note_id, section_id, title, body) VALUES (?, ?, ?, ?)");
    subjects.forEach(s => S.run(s));
    for (const n of notes) {
      N.run(n);
      F.run(n.id, "", n.title, n.lead);
      for (const c of n.sections) { C.run(n.id, c.id, c.position, c.title, JSON.stringify(c.questions)); F.run(n.id, c.id, c.title, c.body); }
    }
  })();
  const totals = { subjects: subjects.length, notes: notes.length, sections: notes.reduce((n, x) => n + x.sections.length, 0),
                   widgets: notes.reduce((n, x) => n + x.widgets, 0), runnable: notes.reduce((n, x) => n + x.runnable, 0), questions: notes.reduce((n, x) => n + x.questions, 0) };
  if (!quiet) console.log("catalogue:", totals);
  return totals;
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, "$1"))) { buildCatalogue(); (await import("./papers.js")).buildPapers(); }
