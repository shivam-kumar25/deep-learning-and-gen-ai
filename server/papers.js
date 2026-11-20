// The exam hall: question papers, attempts and grading.
// Papers come from two places, rebuilt on every index:
//   - server/papers/*.json: past papers (PYQs), transcribed with their code as text
//   - the notes: every answerable question (.pq) in a subject's notes becomes that subject's practice paper
// Questions keep a stable key, so rebuilding never touches anyone's attempts.
// Answers and explanations never reach the browser before the question is checked (practice mode) or the attempt is submitted.
import fs from "node:fs";
import path from "node:path";
import { parse } from "node-html-parser";
import { z } from "zod";
import { db, APP_ROOT, REPO_ROOT, contentFile, tx } from "./db.js";
import { requireUser } from "./auth.js";

const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const codeHtml = c => `<div class="codewrap">${c.label ? `<div class="code-label">${esc(c.label)}</div>` : ""}<pre class="code" data-lang="${esc(c.lang || "text")}">${esc(c.text)}</pre></div>`;

// ------------------------------------------------------------------ building
function fromJson() {
  const dir = path.join(APP_ROOT, "server", "papers");
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith(".json")).sort().map(f => {
    const p = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
    const qs = [];
    const add = (q, parentKey = null) => {
      qs.push({
        key: `${p.id}:${q.id}`, parentKey, number: q.number, type: q.type, marks: q.marks,
        prompt: (q.prompt || "") + (q.code || []).map(codeHtml).join("") + (q.after || ""),
        options: q.options || [], answer: q.answer ?? [], explanation: q.explanation || "", note: q.note || null,
      });
      for (const c of q.children || []) add(c, `${p.id}:${q.id}`);
    };
    p.questions.forEach(q => add(q));
    return { id: p.id, title: p.title, course: p.course || "DL", exam: p.exam || "", source: "pyq", subject_id: null, description: p.description || "", duration: p.duration, questions: qs };
  });
}

function fromNotes() {
  const papers = [];
  for (const s of db.prepare("SELECT id, title FROM subjects ORDER BY position").all()) {
    const qs = [];
    for (const n of db.prepare("SELECT id, path, title FROM notes WHERE subject_id = ? ORDER BY position").all(s.id)) {
      const root = parse(fs.readFileSync(contentFile(n.path), "utf8"));
      root.querySelectorAll(".pq").forEach((el, i) => {
        if (!el.getAttribute("data-answer")) return;
        const type = { multi: "MSQ", num: "NUM", text: "TEXT" }[el.getAttribute("data-type")] || "MCQ";
        const letters = el.getAttribute("data-answer").split(",").map(x => x.trim().toUpperCase());
        const why = el.querySelector(".pq-why");
        if (why) {   // live demos and style blocks only work inside the notes: point there instead
          why.querySelectorAll(".demo").forEach(d => d.replaceWith(`<p class="demo-link">▶ A live demo of this is in the note.</p>`));
          why.querySelectorAll("style, script, textarea, template").forEach(x => x.remove());
        }
        // relative images and links only make sense next to the note: point them at /content and the reader
        const base = "/content/" + path.posix.dirname(n.path.replace(/\\/g, "/")) + "/";
        el.querySelectorAll("[src]").forEach(x => { const s = x.getAttribute("src"); if (!/^(https?:|data:|\/)/.test(s)) x.setAttribute("src", base + s); });
        el.querySelectorAll("a[href]").forEach(x => {
          const h = x.getAttribute("href");
          if (h.startsWith("#")) x.setAttribute("href", `/app/learn/${n.id}${h}`);
          else if (!/^(https?:|mailto:|\/)/.test(h)) x.setAttribute("href", base + h);
          if (/^https?:/.test(h)) { x.setAttribute("target", "_blank"); x.setAttribute("rel", "noopener"); }
        });
        const sec = el.closest("section.sec");
        qs.push({
          key: `notes:${n.id}:${i}`, parentKey: null, number: String(qs.length + 1), type, marks: type === "MSQ" ? 2 : 1,
          prompt: el.querySelector(".pq-q")?.innerHTML || "",
          options: el.querySelectorAll(".pq-opts > li").map(li => li.innerHTML),
          answer: type === "NUM" ? { value: Number(el.getAttribute("data-answer")), tol: Number(el.getAttribute("data-tol") || 0) }
                : type === "TEXT" ? { texts: el.getAttribute("data-answer").split("|").map(x => x.trim()) } : letters.map(l => l.charCodeAt(0) - 65),
          explanation: why ? why.innerHTML : "", note: n.id + (sec?.getAttribute("id") ? "#" + sec.getAttribute("id") : ""),
          from: n.title,
        });
      });
    }
    if (qs.length) papers.push({ id: `notes-${s.id}`, title: `${s.title}: every question in the notes`, course: "DL", exam: "Practice", source: "notes", subject_id: s.id,
      description: `${qs.length} questions collected from the ${s.title} notes, in reading order, each with its explanation and a link back to the section.`, duration: Math.max(10, Math.ceil(qs.length * 1.5)), questions: qs });
  }
  return papers;
}

export function buildPapers({ quiet = false } = {}) {
  const papers = [...fromJson(), ...fromNotes()];
  tx(() => {
    const keepPapers = new Set(papers.map(p => p.id));
    for (const { id } of db.prepare("SELECT id FROM papers").all()) if (!keepPapers.has(id)) db.prepare("DELETE FROM papers WHERE id = ?").run(id);
    const P = db.prepare(`INSERT INTO papers (id, title, course, exam, source, subject_id, description, duration, position) VALUES (@id, @title, @course, @exam, @source, @subject_id, @description, @duration, @position)
                          ON CONFLICT(id) DO UPDATE SET title=excluded.title, course=excluded.course, exam=excluded.exam, source=excluded.source, subject_id=excluded.subject_id, description=excluded.description, duration=excluded.duration, position=excluded.position`);
    const Q = db.prepare(`INSERT INTO paper_questions (key, paper_id, position, number, type, marks, parent_id, prompt, options, answer, explanation, note_ref)
                          VALUES (@key, @paper_id, @position, @number, @type, @marks, @parent_id, @prompt, @options, @answer, @explanation, @note_ref)
                          ON CONFLICT(key) DO UPDATE SET paper_id=excluded.paper_id, position=excluded.position, number=excluded.number, type=excluded.type, marks=excluded.marks, parent_id=excluded.parent_id,
                          prompt=excluded.prompt, options=excluded.options, answer=excluded.answer, explanation=excluded.explanation, note_ref=excluded.note_ref`);
    papers.forEach((p, pi) => {
      P.run({ ...p, position: pi });
      const keys = [];
      p.questions.forEach((q, i) => {
        const parent = q.parentKey ? db.prepare("SELECT id FROM paper_questions WHERE key = ?").get(q.parentKey)?.id ?? null : null;
        Q.run({ key: q.key, paper_id: p.id, position: i, number: q.number, type: q.type, marks: q.marks, parent_id: parent, prompt: q.prompt,
                options: JSON.stringify(q.options), answer: JSON.stringify(q.answer), explanation: q.explanation, note_ref: q.note || null });
        keys.push(q.key);
      });
      const placeholders = keys.map(() => "?").join(",") || "''";
      db.prepare(`DELETE FROM paper_questions WHERE paper_id = ? AND key NOT IN (${placeholders})`).run(p.id, ...keys);
    });
  })();
  const total = papers.reduce((n, p) => n + p.questions.length, 0);
  if (!quiet) console.log(`papers: ${papers.length} (${total} questions)`);
  return { papers: papers.length, questions: total };
}

// ------------------------------------------------------------------ grading
function grade(q, response) {
  if (q.type === "COMP") return { correct: null, marks: 0 };
  const ans = JSON.parse(q.answer);
  if (response == null || (Array.isArray(response) && !response.length) || response === "") return { correct: 0, marks: 0 };
  let ok;
  if (q.type === "NUM") { const v = Number(response); ok = Number.isFinite(v) && Math.abs(v - ans.value) <= (ans.tol || 0); }
  else if (q.type === "TEXT") { const norm = s => String(s).trim().replace(/\s+/g, " ").toLowerCase(); ok = ans.texts.some(t => norm(t) === norm(response)); }
  else { const r = [...new Set(response.map(Number))].sort(), a = [...ans].sort(); ok = r.length === a.length && r.every((x, i) => x === a[i]); }
  return { correct: ok ? 1 : 0, marks: ok ? q.marks : 0 };
}
const publicQ = (q, reveal) => ({
  id: q.id, number: q.number, type: q.type, marks: q.marks, parent_id: q.parent_id, prompt: q.prompt, options: JSON.parse(q.options),
  ...(reveal ? { answer: JSON.parse(q.answer), explanation: q.explanation, note_ref: q.note_ref } : {}),
});
const deadlinePassed = a => a.mode === "exam" && a.deadline && Date.now() > new Date(a.deadline).getTime() + 15_000;

function attemptView(a) {
  const qs = db.prepare("SELECT * FROM paper_questions WHERE paper_id = ? ORDER BY position").all(a.paper_id);
  const answers = db.prepare("SELECT question_id, response, flagged, visited, seconds, checked, correct, marks FROM attempt_answers WHERE attempt_id = ?").all(a.id)
    .map(r => ({ ...r, response: r.response == null ? null : JSON.parse(r.response) }));
  const byQ = Object.fromEntries(answers.map(r => [r.question_id, r]));
  const done = !!a.submitted_at;
  return {
    attempt: { id: a.id, paper_id: a.paper_id, mode: a.mode, started_at: a.started_at, deadline: a.deadline, submitted_at: a.submitted_at, score: a.score, max_score: a.max_score, now: new Date().toISOString() },
    paper: db.prepare("SELECT id, title, exam, course, source, duration, description FROM papers WHERE id = ?").get(a.paper_id),
    questions: qs.map(q => publicQ(q, done || byQ[q.id]?.checked)),
    answers,
  };
}

// ------------------------------------------------------------------ routes
export function routes(app) {
  app.get("/api/papers", requireUser, (req, res) => {
    const papers = db.prepare(`SELECT p.*, (SELECT COUNT(*) FROM paper_questions q WHERE q.paper_id = p.id AND q.type != 'COMP') questions,
                                      (SELECT COALESCE(SUM(marks), 0) FROM paper_questions q WHERE q.paper_id = p.id) max_marks
                               FROM papers p ORDER BY p.position`).all();
    const mine = db.prepare(`SELECT paper_id, COUNT(*) attempts, MAX(CASE WHEN submitted_at IS NOT NULL THEN score END) best,
                                    MAX(CASE WHEN submitted_at IS NULL THEN id END) open_id
                             FROM attempts WHERE user_id = ? GROUP BY paper_id`).all(req.user.id);
    const m = Object.fromEntries(mine.map(r => [r.paper_id, r]));
    const subj = Object.fromEntries(db.prepare("SELECT id, hue, icon FROM subjects").all().map(s => [s.id, s]));
    res.json({ papers: papers.map(p => ({ ...p, hue: subj[p.subject_id]?.hue ?? 250, icon: subj[p.subject_id]?.icon ?? "ScrollText", me: m[p.id] || null })) });
  });

  app.get("/api/papers/:id/attempts", requireUser, (req, res) => {
    res.json({ attempts: db.prepare("SELECT id, mode, started_at, submitted_at, score, max_score FROM attempts WHERE user_id = ? AND paper_id = ? ORDER BY id DESC").all(req.user.id, req.params.id) });
  });

  const Start = z.object({ mode: z.enum(["exam", "practice"]), fresh: z.boolean().optional() });
  app.post("/api/papers/:id/start", requireUser, (req, res) => {
    const r = Start.safeParse(req.body); if (!r.success) return res.status(400).json({ error: "Invalid request." });
    const paper = db.prepare("SELECT * FROM papers WHERE id = ?").get(req.params.id);
    if (!paper) return res.status(404).json({ error: "Paper not found." });
    let a = !r.data.fresh && db.prepare("SELECT * FROM attempts WHERE user_id = ? AND paper_id = ? AND mode = ? AND submitted_at IS NULL ORDER BY id DESC").get(req.user.id, paper.id, r.data.mode);
    if (a && deadlinePassed(a)) { submit(a); a = null; }
    if (!a) {
      const deadline = r.data.mode === "exam" ? new Date(Date.now() + paper.duration * 60_000).toISOString() : null;
      const { lastInsertRowid } = db.prepare("INSERT INTO attempts (user_id, paper_id, mode, deadline) VALUES (?, ?, ?, ?)").run(req.user.id, paper.id, r.data.mode, deadline);
      a = db.prepare("SELECT * FROM attempts WHERE id = ?").get(lastInsertRowid);
    }
    res.json(attemptView(a));
  });

  const own = (req, res) => {
    const a = db.prepare("SELECT * FROM attempts WHERE id = ? AND user_id = ?").get(Number(req.params.id), req.user.id);
    if (!a) { res.status(404).json({ error: "Attempt not found." }); return null; }
    return a;
  };
  app.get("/api/attempts/:id", requireUser, (req, res) => {
    let a = own(req, res); if (!a) return;
    if (!a.submitted_at && deadlinePassed(a)) a = submit(a);
    res.json(attemptView(a));
  });

  const Save = z.object({ response: z.union([z.array(z.number().int().min(0).max(20)).max(20), z.string().max(40), z.null()]).optional(), flagged: z.boolean().optional(), seconds: z.number().int().min(0).max(3600).optional() });
  app.put("/api/attempts/:id/answers/:qid", requireUser, (req, res) => {
    const a = own(req, res); if (!a) return;
    if (a.submitted_at) return res.status(409).json({ error: "This attempt is already submitted." });
    if (deadlinePassed(a)) { submit(a); return res.status(409).json({ error: "Time is up: the paper was submitted." }); }
    const r = Save.safeParse(req.body); if (!r.success) return res.status(400).json({ error: "Invalid answer." });
    const q = db.prepare("SELECT id, type FROM paper_questions WHERE id = ? AND paper_id = ?").get(Number(req.params.qid), a.paper_id);
    if (!q || q.type === "COMP") return res.status(404).json({ error: "Question not found." });
    const cur = db.prepare("SELECT * FROM attempt_answers WHERE attempt_id = ? AND question_id = ?").get(a.id, q.id);
    if (cur?.checked && r.data.response !== undefined) return res.status(409).json({ error: "This answer has been checked already." });
    db.prepare(`INSERT INTO attempt_answers (attempt_id, question_id, response, flagged, seconds) VALUES (@a, @q, @resp, @flag, @sec)
                ON CONFLICT DO UPDATE SET response = CASE WHEN @hasResp THEN @resp ELSE response END,
                flagged = CASE WHEN @hasFlag THEN @flag ELSE flagged END, seconds = seconds + @sec, visited = 1`)
      .run({ a: a.id, q: q.id, resp: r.data.response === undefined ? null : JSON.stringify(r.data.response), hasResp: r.data.response !== undefined ? 1 : 0,
             flag: r.data.flagged ? 1 : 0, hasFlag: r.data.flagged !== undefined ? 1 : 0, sec: r.data.seconds || 0 });
    res.json({ ok: true });
  });

  app.post("/api/attempts/:id/check/:qid", requireUser, (req, res) => {
    const a = own(req, res); if (!a) return;
    if (a.mode !== "practice" || a.submitted_at) return res.status(409).json({ error: "Checking one question is only for practice mode." });
    const q = db.prepare("SELECT * FROM paper_questions WHERE id = ? AND paper_id = ?").get(Number(req.params.qid), a.paper_id);
    if (!q || q.type === "COMP") return res.status(404).json({ error: "Question not found." });
    const cur = db.prepare("SELECT response FROM attempt_answers WHERE attempt_id = ? AND question_id = ?").get(a.id, q.id);
    const g = grade(q, cur?.response == null ? null : JSON.parse(cur.response));
    db.prepare(`INSERT INTO attempt_answers (attempt_id, question_id, checked, correct, marks) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT DO UPDATE SET checked = 1, correct = excluded.correct, marks = excluded.marks`).run(a.id, q.id, g.correct, g.marks);
    res.json({ ...g, question: publicQ(q, true) });
  });

  function submit(a) {
    const qs = db.prepare("SELECT * FROM paper_questions WHERE paper_id = ?").all(a.paper_id);
    const resp = Object.fromEntries(db.prepare("SELECT question_id, response FROM attempt_answers WHERE attempt_id = ?").all(a.id).map(r => [r.question_id, r.response == null ? null : JSON.parse(r.response)]));
    let score = 0, max = 0;
    tx(() => {
      for (const q of qs) {
        if (q.type === "COMP") continue;
        const g = grade(q, resp[q.id]); score += g.marks; max += q.marks;
        db.prepare(`INSERT INTO attempt_answers (attempt_id, question_id, correct, marks, visited) VALUES (?, ?, ?, ?, 0)
                    ON CONFLICT DO UPDATE SET correct = excluded.correct, marks = excluded.marks`).run(a.id, q.id, g.correct, g.marks);
      }
      db.prepare("UPDATE attempts SET submitted_at = datetime('now'), score = ?, max_score = ? WHERE id = ?").run(score, max, a.id);
    })();
    return db.prepare("SELECT * FROM attempts WHERE id = ?").get(a.id);
  }
  app.post("/api/attempts/:id/submit", requireUser, (req, res) => {
    let a = own(req, res); if (!a) return;
    if (!a.submitted_at) a = submit(a);
    res.json(attemptView(a));
  });

  app.delete("/api/attempts/:id", requireUser, (req, res) => {
    const a = own(req, res); if (!a) return;
    db.prepare("DELETE FROM attempts WHERE id = ?").run(a.id);
    res.json({ ok: true });
  });
}
