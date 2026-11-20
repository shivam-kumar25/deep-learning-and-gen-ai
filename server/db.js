// SQLite through better-sqlite3: one file, real SQL, foreign keys, WAL, FTS5 search.
import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const APP_ROOT = path.resolve(HERE, "..");                 // the repo root: the app lives here
export const REPO_ROOT = path.join(APP_ROOT, "notes");             // the HTML notes it serves: /content/<path> is read from notes/
const CONTENT_ROOTS = {};
export function contentFile(rel) {
  const [top, ...rest] = String(rel).split(/[\\/]/);
  const base = CONTENT_ROOTS[top] ? CONTENT_ROOTS[top] : REPO_ROOT;
  const file = path.resolve(base, CONTENT_ROOTS[top] ? rest.join("/") : String(rel));
  return file.startsWith(base + path.sep) ? file : null;     // null: outside the notes (traversal)
}

const file = process.env.DB_FILE || path.join(APP_ROOT, "data", "studyloom.db");
if (file !== ":memory:") fs.mkdirSync(path.dirname(file), { recursive: true });

export const db = new Database(file);
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");
db.pragma("busy_timeout = 5000");

// Ordered, append-only migrations. Never edit one that has shipped: add a new one.
const MIGRATIONS = [
  `CREATE TABLE users (
     id            INTEGER PRIMARY KEY,
     email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
     name          TEXT NOT NULL,
     password_hash TEXT NOT NULL,
     theme         TEXT NOT NULL DEFAULT 'system' CHECK (theme IN ('system','light','dark')),
     daily_goal    INTEGER NOT NULL DEFAULT 30 CHECK (daily_goal BETWEEN 5 AND 600),
     created_at    TEXT NOT NULL DEFAULT (datetime('now'))
   );
   CREATE TABLE sessions (
     id          TEXT PRIMARY KEY,                -- sha256 of the cookie token (the token itself is never stored)
     user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     created_at  TEXT NOT NULL DEFAULT (datetime('now')),
     expires_at  TEXT NOT NULL,
     user_agent  TEXT
   );
   CREATE INDEX sessions_user ON sessions(user_id);

   -- the catalogue, rebuilt by the indexer from the HTML notes
   CREATE TABLE subjects (
     id          TEXT PRIMARY KEY,                -- folder name, e.g. 03-regression
     num         TEXT NOT NULL,
     title       TEXT NOT NULL,
     blurb       TEXT NOT NULL DEFAULT '',
     hue         INTEGER NOT NULL,                -- colour identity (0-360)
     icon        TEXT NOT NULL,
     index_path  TEXT,
     position    INTEGER NOT NULL
   );
   CREATE TABLE notes (
     id          TEXT PRIMARY KEY,                -- path without .html, e.g. 03-regression/01_linear_regression
     subject_id  TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
     path        TEXT NOT NULL UNIQUE,            -- repo-relative .html path
     title       TEXT NOT NULL,
     eyebrow     TEXT NOT NULL DEFAULT '',
     lead        TEXT NOT NULL DEFAULT '',
     kind        TEXT NOT NULL DEFAULT 'note',    -- note | app | topic
     group_title TEXT NOT NULL DEFAULT '',
     position    INTEGER NOT NULL,
     minutes     INTEGER NOT NULL DEFAULT 5,
     widgets     INTEGER NOT NULL DEFAULT 0,
     runnable    INTEGER NOT NULL DEFAULT 0,
     questions   INTEGER NOT NULL DEFAULT 0
   );
   CREATE INDEX notes_subject ON notes(subject_id, position);
   CREATE TABLE sections (
     note_id     TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
     id          TEXT NOT NULL,                   -- the section element's id in the page
     position    INTEGER NOT NULL,
     title       TEXT NOT NULL,
     PRIMARY KEY (note_id, id)
   );
   CREATE VIRTUAL TABLE search USING fts5(note_id UNINDEXED, section_id UNINDEXED, title, body, tokenize = 'porter unicode61');

   -- per user
   CREATE TABLE section_progress (
     user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     note_id     TEXT NOT NULL,
     section_id  TEXT NOT NULL,
     done_at     TEXT NOT NULL DEFAULT (datetime('now')),
     PRIMARY KEY (user_id, note_id, section_id)
   );
   CREATE TABLE reading_state (
     user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     note_id      TEXT NOT NULL,
     last_section TEXT,
     opened_at    TEXT NOT NULL DEFAULT (datetime('now')),
     opens        INTEGER NOT NULL DEFAULT 1,
     PRIMARY KEY (user_id, note_id)
   );
   CREATE TABLE activity (
     user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     day         TEXT NOT NULL,                   -- YYYY-MM-DD in the user's time zone
     seconds     INTEGER NOT NULL DEFAULT 0,
     sections    INTEGER NOT NULL DEFAULT 0,
     PRIMARY KEY (user_id, day)
   );
   CREATE TABLE bookmarks (
     user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     note_id     TEXT NOT NULL,
     section_id  TEXT NOT NULL DEFAULT '',
     created_at  TEXT NOT NULL DEFAULT (datetime('now')),
     PRIMARY KEY (user_id, note_id, section_id)
   );
   CREATE TABLE user_notes (
     id          INTEGER PRIMARY KEY,
     user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     note_id     TEXT NOT NULL,
     section_id  TEXT NOT NULL DEFAULT '',
     body        TEXT NOT NULL CHECK (length(body) <= 20000),
     created_at  TEXT NOT NULL DEFAULT (datetime('now')),
     updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
   );
   CREATE INDEX user_notes_note ON user_notes(user_id, note_id);`,

  // v2: the exam hall: question papers, their questions, and each user's attempts
  `CREATE TABLE papers (
     id          TEXT PRIMARY KEY,                -- slug
     title       TEXT NOT NULL,
     course      TEXT NOT NULL DEFAULT 'MLP',
     exam        TEXT NOT NULL DEFAULT '',        -- Quiz 1, Quiz 2, End term, Practice…
     source      TEXT NOT NULL,                   -- pyq | notes
     subject_id  TEXT,                            -- for papers built from a subject's notes
     description TEXT NOT NULL DEFAULT '',
     duration    INTEGER NOT NULL,                -- minutes
     position    INTEGER NOT NULL
   );
   CREATE TABLE paper_questions (
     id          INTEGER PRIMARY KEY,
     key         TEXT NOT NULL UNIQUE,            -- stable across rebuilds (paper id + question id), so attempts survive a re-index
     paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
     position    INTEGER NOT NULL,
     number      TEXT NOT NULL,                   -- shown label: "3", "15", "16"
     type        TEXT NOT NULL CHECK (type IN ('MCQ','MSQ','NUM','TEXT','COMP')),
     marks       REAL NOT NULL,
     parent_id   INTEGER REFERENCES paper_questions(id) ON DELETE CASCADE,   -- sub-question of a comprehension
     prompt      TEXT NOT NULL,                   -- HTML
     options     TEXT NOT NULL DEFAULT '[]',      -- JSON array of HTML strings
     answer      TEXT NOT NULL DEFAULT '[]',      -- JSON: option indexes, or {"value": n, "tol": t} for NUM
     explanation TEXT NOT NULL DEFAULT '',        -- HTML
     note_ref    TEXT                             -- note id (#section) where this is taught
   );
   CREATE INDEX paper_questions_paper ON paper_questions(paper_id, position);
   CREATE TABLE attempts (
     id           INTEGER PRIMARY KEY,
     user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     paper_id     TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
     mode         TEXT NOT NULL CHECK (mode IN ('exam','practice')),
     started_at   TEXT NOT NULL DEFAULT (datetime('now')),
     deadline     TEXT,                            -- exam mode only
     submitted_at TEXT,
     score        REAL,
     max_score    REAL
   );
   CREATE INDEX attempts_user ON attempts(user_id, paper_id);
   CREATE TABLE attempt_answers (
     attempt_id   INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
     question_id  INTEGER NOT NULL REFERENCES paper_questions(id) ON DELETE CASCADE,
     response     TEXT,                            -- JSON: [indexes] or "number"
     flagged      INTEGER NOT NULL DEFAULT 0,
     visited      INTEGER NOT NULL DEFAULT 1,
     seconds      INTEGER NOT NULL DEFAULT 0,
     checked      INTEGER NOT NULL DEFAULT 0,      -- practice mode: answer revealed
     correct      INTEGER,
     marks        REAL,
     PRIMARY KEY (attempt_id, question_id)
   );`,

  // v3: the questions inside each section, so the reader's index can list them under it
  `ALTER TABLE sections ADD COLUMN questions TEXT NOT NULL DEFAULT '[]';   -- JSON: [{ n, text }] (n = the question's number on the page)`,

  // v4: subjects belong to a track (shown in this order in the library)
  `ALTER TABLE subjects ADD COLUMN track TEXT NOT NULL DEFAULT 'theory';`,
];

db.exec("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)");
const current = db.prepare("SELECT version FROM schema_version").get()?.version ?? 0;
if (current === 0 && !db.prepare("SELECT 1 FROM schema_version").get()) db.prepare("INSERT INTO schema_version VALUES (0)").run();
for (let v = current; v < MIGRATIONS.length; v++) {
  db.transaction(() => { db.exec(MIGRATIONS[v]); db.prepare("UPDATE schema_version SET version = ?").run(v + 1); })();
}

export const tx = fn => db.transaction(fn);
