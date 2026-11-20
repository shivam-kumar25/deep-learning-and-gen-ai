// Write the course's assignments and past paper as exam-hall papers (server/papers/*.json), taking every question from the built
// notes by its tag, so the notes and the exam hall never disagree. Marks, case texts (comprehensions) and missing questions come from PAPERS below.
// Run: node tools/notes/build/make_papers.mjs   (after building the notes)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "node-html-parser";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const NOTES = path.join(ROOT, "notes"), OUT = path.join(ROOT, "server", "papers");
const t = s => `<span class="tex">${s}</span>`;
const table = (head, rows) => `<div class="tblwrap"><table><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;

const FOOD = table(["Country", "Home cooking as a % of total food expenses", "Average spending on food expenses per month (in $)"],
  [["US", "6.4", "500"], ["India", "35.8", "200"], ["Russia", "42", "300"], ["Pakistan", "29.1", "100"], ["China", "33.9", "300"]]);
const INCOME = table(["", "Income level 1", "Income level 2"], [["Income", "Rs 15000", "Rs 25000"], ["Qty demanded", "60 units", "90 units"], ["Price per qty", "Rs 40", "Rs 38"]]);

const PAPERS = [];   // filled in as question papers arrive (see the BDM repo for the format)

// ------------------------------------------------------------------ read every tagged question from the built notes
const files = fs.readdirSync(NOTES).filter(d => /^b\d/.test(d)).flatMap(d => fs.readdirSync(path.join(NOTES, d)).filter(f => /^\d\d_.*\.html$/.test(f)).map(f => `${d}/${f}`));
const found = [];
for (const rel of files) {
  const root = parse(fs.readFileSync(path.join(NOTES, rel), "utf8"));
  const noteId = rel.replace(/\.html$/, "");
  for (const el of root.querySelectorAll(".pq")) {
    const tag = el.getAttribute("data-tag") || "", m = tag.match(/^(.*) · Q(\d+)$/);
    if (!m) continue;
    const sec = el.closest("section.sec");
    const type = { single: "MCQ", multi: "MSQ", num: "NUM", text: "TEXT" }[el.getAttribute("data-type")] || "MCQ";
    const ans = el.getAttribute("data-answer");
    el.querySelectorAll(".pq-note").forEach(n => n.remove());
    found.push({
      tag: m[1], n: Number(m[2]), type,
      prompt: el.querySelector(".pq-q").innerHTML.trim(),
      options: el.querySelectorAll(".pq-opts > li").map(li => li.innerHTML.trim()),
      answer: type === "NUM" ? { value: Number(ans), tol: Number(el.getAttribute("data-tol") || 0) }
            : type === "TEXT" ? { texts: ans.split("|").map(s => s.trim()) }
            : ans.split(",").map(s => s.trim().charCodeAt(0) - 65),
      explanation: (el.querySelector(".pq-why")?.innerHTML || "").trim(),
      note: noteId + (sec?.getAttribute("id") ? "#" + sec.getAttribute("id") : ""),
    });
  }
}

fs.mkdirSync(OUT, { recursive: true });
let used = 0;
for (const P of PAPERS) {
  const qs = found.filter(q => q.tag === P.tag).sort((a, b) => a.n - b.n);
  if (!qs.length) throw new Error("no questions tagged " + P.tag);
  const dup = qs.find((q, i) => i && q.n === qs[i - 1].n); if (dup) throw new Error(`${P.tag}: Q${dup.n} appears twice`);
  const inComp = new Set((P.comps || []).flatMap(c => c.children));
  const mk = q => ({ id: `q${q.n}`, number: String(q.n), type: q.type, marks: P.marks(q.n), prompt: q.prompt, options: q.options, answer: q.answer, explanation: q.explanation, note: q.note });
  const out = [];
  for (const q of qs) {
    for (const c of P.comps || []) if (c.before === q.n) out.push({ id: `c${c.number.replace(/\W+/g, "")}`, number: c.number, type: "COMP", marks: 0, prompt: c.prompt, options: [], answer: [], explanation: "", note: null,
      children: c.children.map(n => { const x = qs.find(y => y.n === n); if (!x) throw new Error(`${P.tag}: case child Q${n} missing`); return mk(x); }) });
    if (!inComp.has(q.n)) out.push(mk(q));
  }
  const n = qs.length, marks = qs.reduce((s, q) => s + P.marks(q.n), 0);
  const paper = { id: P.id, title: P.title, course: "BDM", exam: P.exam, duration: P.duration, description: P.description.replace("{n}", n).replace("{marks}", marks), questions: out };
  fs.writeFileSync(path.join(OUT, P.id + ".json"), JSON.stringify(paper, null, 1) + "\n");
  used += n;
  console.log(`${P.id}: ${n} questions, ${marks} marks`);
}
const tags = new Set(PAPERS.map(p => p.tag)), stray = found.filter(q => !tags.has(q.tag));
console.log(`questions placed in papers: ${used} of ${found.length}` + (stray.length ? `; untagged for a paper: ${stray.map(q => q.tag + " Q" + q.n).join(", ")}` : ""));
