import { z } from "zod";
import { db, tx } from "./db.js";
import { requireUser } from "./auth.js";

const Day = z.string().regex(/^\d{4}-\d{2}-\d{2}$/);
const bad = (res, msg = "Invalid request.") => res.status(400).json({ error: msg });
const noteExists = id => !!db.prepare("SELECT 1 FROM notes WHERE id = ?").get(id);

export function routes(app) {
  // ---------------------------------------------------------------- public
  app.get("/api/stats", (req, res) => {
    const q = db.prepare(`SELECT COUNT(*) notes, SUM(widgets) widgets, SUM(runnable) runnable, SUM(questions) questions,
                                 (SELECT COUNT(*) FROM sections) sections, (SELECT COUNT(*) FROM subjects) subjects,
                                 SUM(kind = 'lab') labs, SUM(minutes) minutes FROM notes`).get();
    res.json(q);
  });

  app.get("/api/catalogue", (req, res) => {
    const subjects = db.prepare("SELECT * FROM subjects ORDER BY position").all();
    const notes = db.prepare("SELECT id, subject_id, path, title, eyebrow, lead, kind, group_title, position, minutes, widgets, runnable, questions, (SELECT COUNT(*) FROM sections s WHERE s.note_id = notes.id) sections FROM notes ORDER BY subject_id, position").all();
    let done = {};
    if (req.user) for (const r of db.prepare("SELECT note_id, COUNT(*) n FROM section_progress WHERE user_id = ? GROUP BY note_id").all(req.user.id)) done[r.note_id] = r.n;
    res.json({ subjects, notes: notes.map(n => ({ ...n, done: done[n.id] || 0 })) });
  });

  app.get("/api/notes/:subject/*rest", (req, res) => {
    const id = `${req.params.subject}/${[].concat(req.params.rest).join("/")}`;
    const note = db.prepare("SELECT * FROM notes WHERE id = ?").get(id);
    if (!note) return res.status(404).json({ error: "Note not found." });
    const sections = db.prepare("SELECT id, title, position, questions FROM sections WHERE note_id = ? ORDER BY position").all(id).map(s => ({ ...s, questions: JSON.parse(s.questions) }));
    const siblings = db.prepare("SELECT id, title, position, group_title FROM notes WHERE subject_id = ? ORDER BY position").all(note.subject_id);
    const i = siblings.findIndex(n => n.id === id);
    const subject = db.prepare("SELECT * FROM subjects WHERE id = ?").get(note.subject_id);
    let me = null;
    if (req.user) {
      const uid = req.user.id;
      me = {
        done: db.prepare("SELECT section_id FROM section_progress WHERE user_id = ? AND note_id = ?").all(uid, id).map(r => r.section_id),
        state: db.prepare("SELECT last_section, opened_at, opens FROM reading_state WHERE user_id = ? AND note_id = ?").get(uid, id) || null,
        bookmarks: db.prepare("SELECT section_id FROM bookmarks WHERE user_id = ? AND note_id = ?").all(uid, id).map(r => r.section_id),
        notes: db.prepare("SELECT id, section_id, body, created_at, updated_at FROM user_notes WHERE user_id = ? AND note_id = ? ORDER BY created_at").all(uid, id),
      };
    }
    res.json({ note, subject, sections, prev: siblings[i - 1] || null, next: siblings[i + 1] || null, me });
  });

  app.get("/api/search", (req, res) => {
    const q = String(req.query.q || "").trim().slice(0, 100);
    if (q.length < 2) return res.json({ results: [] });
    // every word as a prefix term, quoted so user input can't inject FTS syntax
    const terms = q.split(/\s+/).filter(Boolean).slice(0, 8).map(w => `"${w.replace(/"/g, "")}"*`).join(" ");
    try {
      const rows = db.prepare(`SELECT s.note_id, s.section_id, n.title note_title, n.subject_id, sub.title subject_title, sub.hue,
                                      COALESCE(sec.title, '') section_title,
                                      snippet(search, 3, char(2), char(3), '…', 14) snippet   -- … around matches; the client renders them as <mark> (no HTML from the notes is ever injected), bm25(search, 0, 0, 6, 1) rank
                               FROM search s JOIN notes n ON n.id = s.note_id JOIN subjects sub ON sub.id = n.subject_id
                               LEFT JOIN sections sec ON sec.note_id = s.note_id AND sec.id = s.section_id
                               WHERE search MATCH ? ORDER BY rank LIMIT 30`).all(terms);
      res.json({ results: rows });
    } catch { res.json({ results: [] }); }
  });

  // ---------------------------------------------------------------- signed in
  app.get("/api/dashboard", requireUser, (req, res) => {
    const uid = req.user.id;
    const today = Day.safeParse(req.query.today).success ? req.query.today : new Date().toISOString().slice(0, 10);
    const recent = db.prepare(`SELECT r.note_id, r.last_section, r.opened_at, n.title, n.subject_id, n.minutes, sub.title subject_title, sub.hue, sub.icon,
                                      (SELECT COUNT(*) FROM sections s WHERE s.note_id = n.id) sections,
                                      (SELECT COUNT(*) FROM section_progress p WHERE p.user_id = r.user_id AND p.note_id = n.id) done
                               FROM reading_state r JOIN notes n ON n.id = r.note_id JOIN subjects sub ON sub.id = n.subject_id
                               WHERE r.user_id = ? ORDER BY r.opened_at DESC LIMIT 6`).all(uid);
    const subjects = db.prepare(`SELECT sub.id, sub.title, sub.num, sub.hue, sub.icon,
                                        (SELECT COUNT(*) FROM sections s JOIN notes n ON n.id = s.note_id WHERE n.subject_id = sub.id) total,
                                        (SELECT COUNT(*) FROM section_progress p JOIN notes n ON n.id = p.note_id WHERE p.user_id = ? AND n.subject_id = sub.id) done
                                 FROM subjects sub ORDER BY sub.position`).all(uid);
    const days = db.prepare("SELECT day, seconds, sections FROM activity WHERE user_id = ? AND day > date(?, '-182 days') ORDER BY day").all(uid, today);
    // streak: consecutive days up to today (or yesterday) with at least one minute or one section
    const active = new Set(days.filter(d => d.seconds >= 60 || d.sections > 0).map(d => d.day));
    let streak = 0, d = new Date(today + "T12:00:00Z");
    if (!active.has(today)) d.setUTCDate(d.getUTCDate() - 1);
    while (active.has(d.toISOString().slice(0, 10))) { streak++; d.setUTCDate(d.getUTCDate() - 1); }
    const todayRow = days.find(x => x.day === today) || { seconds: 0, sections: 0 };
    // what to read next: the first unfinished note in the most recently opened subject, else the very first note
    const lastSubject = recent[0]?.subject_id;
    const next = db.prepare(`SELECT n.id, n.title, n.minutes, n.subject_id, sub.title subject_title, sub.hue, sub.icon
                             FROM notes n JOIN subjects sub ON sub.id = n.subject_id
                             WHERE (SELECT COUNT(*) FROM section_progress p WHERE p.user_id = ? AND p.note_id = n.id) < (SELECT COUNT(*) FROM sections s WHERE s.note_id = n.id)
                             ORDER BY (n.subject_id = ?) DESC, sub.position, n.position LIMIT 1`).get(uid, lastSubject || "");
    const totals = db.prepare("SELECT COUNT(*) sections, COUNT(DISTINCT note_id) notes FROM section_progress WHERE user_id = ?").get(uid);
    const bookmarks = db.prepare(`SELECT b.note_id, b.section_id, b.created_at, n.title, sec.title section_title, sub.hue FROM bookmarks b JOIN notes n ON n.id = b.note_id
                                  JOIN subjects sub ON sub.id = n.subject_id LEFT JOIN sections sec ON sec.note_id = b.note_id AND sec.id = b.section_id
                                  WHERE b.user_id = ? ORDER BY b.created_at DESC LIMIT 8`).all(uid);
    res.json({ recent, subjects, days, streak, today: { ...todayRow, goal: req.user.daily_goal * 60 }, next, totals, bookmarks });
  });

  const Done = z.object({ noteId: z.string().max(300), sectionId: z.string().max(200), done: z.boolean(), today: Day });
  app.post("/api/progress", requireUser, (req, res) => {
    const r = Done.safeParse(req.body); if (!r.success || !noteExists(r.data.noteId)) return bad(res);
    const { noteId, sectionId, done, today } = r.data, uid = req.user.id;
    tx(() => {
      if (done) {
        const c = db.prepare("INSERT OR IGNORE INTO section_progress (user_id, note_id, section_id) VALUES (?, ?, ?)").run(uid, noteId, sectionId).changes;
        if (c) db.prepare("INSERT INTO activity (user_id, day, sections) VALUES (?, ?, 1) ON CONFLICT DO UPDATE SET sections = sections + 1").run(uid, today);
      } else db.prepare("DELETE FROM section_progress WHERE user_id = ? AND note_id = ? AND section_id = ?").run(uid, noteId, sectionId);
    })();
    res.json({ done: db.prepare("SELECT section_id FROM section_progress WHERE user_id = ? AND note_id = ?").all(uid, noteId).map(x => x.section_id) });
  });

  const Open = z.object({ noteId: z.string().max(300), section: z.string().max(200).nullable().optional() });
  app.post("/api/reading", requireUser, (req, res) => {
    const r = Open.safeParse(req.body); if (!r.success || !noteExists(r.data.noteId)) return bad(res);
    db.prepare(`INSERT INTO reading_state (user_id, note_id, last_section) VALUES (?, ?, ?)
                ON CONFLICT DO UPDATE SET last_section = COALESCE(excluded.last_section, last_section), opened_at = datetime('now'),
                opens = opens + (CASE WHEN excluded.last_section IS NULL THEN 1 ELSE 0 END)`).run(req.user.id, r.data.noteId, r.data.section ?? null);
    res.json({ ok: true });
  });

  // study time: the reader sends a beat every 30 s while the tab is visible and the user is active
  const Beat = z.object({ seconds: z.number().int().min(1).max(120), today: Day });
  app.post("/api/heartbeat", requireUser, (req, res) => {
    const r = Beat.safeParse(req.body); if (!r.success) return bad(res);
    db.prepare("INSERT INTO activity (user_id, day, seconds) VALUES (?, ?, ?) ON CONFLICT DO UPDATE SET seconds = seconds + excluded.seconds").run(req.user.id, r.data.today, r.data.seconds);
    res.json({ ok: true });
  });

  const Mark = z.object({ noteId: z.string().max(300), sectionId: z.string().max(200).default(""), on: z.boolean() });
  app.post("/api/bookmarks", requireUser, (req, res) => {
    const r = Mark.safeParse(req.body); if (!r.success || !noteExists(r.data.noteId)) return bad(res);
    const { noteId, sectionId, on } = r.data;
    if (on) db.prepare("INSERT OR IGNORE INTO bookmarks (user_id, note_id, section_id) VALUES (?, ?, ?)").run(req.user.id, noteId, sectionId);
    else db.prepare("DELETE FROM bookmarks WHERE user_id = ? AND note_id = ? AND section_id = ?").run(req.user.id, noteId, sectionId);
    res.json({ ok: true });
  });
  app.get("/api/bookmarks", requireUser, (req, res) => {
    res.json({ bookmarks: db.prepare(`SELECT b.note_id, b.section_id, b.created_at, n.title, sec.title section_title, sub.title subject_title, sub.hue
                                      FROM bookmarks b JOIN notes n ON n.id = b.note_id JOIN subjects sub ON sub.id = n.subject_id
                                      LEFT JOIN sections sec ON sec.note_id = b.note_id AND sec.id = b.section_id WHERE b.user_id = ? ORDER BY b.created_at DESC`).all(req.user.id) });
  });

  const NoteIn = z.object({ noteId: z.string().max(300), sectionId: z.string().max(200).default(""), body: z.string().trim().min(1, "Write something first.").max(20000) });
  app.post("/api/my-notes", requireUser, (req, res) => {
    const r = NoteIn.safeParse(req.body); if (!r.success || !noteExists(r.data.noteId)) return bad(res, r.error?.issues[0]?.message);
    const { lastInsertRowid } = db.prepare("INSERT INTO user_notes (user_id, note_id, section_id, body) VALUES (?, ?, ?, ?)").run(req.user.id, r.data.noteId, r.data.sectionId, r.data.body);
    res.status(201).json({ note: db.prepare("SELECT id, section_id, body, created_at, updated_at FROM user_notes WHERE id = ?").get(lastInsertRowid) });
  });
  app.patch("/api/my-notes/:id", requireUser, (req, res) => {
    const body = z.string().trim().min(1).max(20000).safeParse(req.body?.body); if (!body.success) return bad(res);
    const c = db.prepare("UPDATE user_notes SET body = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?").run(body.data, Number(req.params.id), req.user.id).changes;
    if (!c) return res.status(404).json({ error: "Not found." });
    res.json({ note: db.prepare("SELECT id, section_id, body, created_at, updated_at FROM user_notes WHERE id = ?").get(Number(req.params.id)) });
  });
  app.delete("/api/my-notes/:id", requireUser, (req, res) => {
    db.prepare("DELETE FROM user_notes WHERE id = ? AND user_id = ?").run(Number(req.params.id), req.user.id);
    res.json({ ok: true });
  });
  app.get("/api/my-notes", requireUser, (req, res) => {
    res.json({ notes: db.prepare(`SELECT u.id, u.note_id, u.section_id, u.body, u.updated_at, n.title, sec.title section_title, sub.title subject_title, sub.hue
                                  FROM user_notes u JOIN notes n ON n.id = u.note_id JOIN subjects sub ON sub.id = n.subject_id
                                  LEFT JOIN sections sec ON sec.note_id = u.note_id AND sec.id = u.section_id WHERE u.user_id = ? ORDER BY u.updated_at DESC`).all(req.user.id) });
  });

  // everything we store about you, as JSON (data portability)
  app.get("/api/account/export", requireUser, (req, res) => {
    const uid = req.user.id, all = t => db.prepare(`SELECT * FROM ${t} WHERE user_id = ?`).all(uid);
    res.set("Content-Disposition", 'attachment; filename="studyloom-export.json"');
    res.json({ account: { email: req.user.email, name: req.user.name, created_at: req.user.created_at }, progress: all("section_progress"), reading: all("reading_state"),
               activity: all("activity"), bookmarks: all("bookmarks"), notes: all("user_notes") });
  });
}
