// node --test server/tests/   (an in-memory database; the catalogue is built from the real notes)
process.env.NODE_ENV = "test";
process.env.DB_FILE = ":memory:";
process.env.SL_LIMIT_SCALE = "5";   // the suite creates more accounts than one person would in a minute
import { test, before, after } from "node:test";
import assert from "node:assert/strict";

let base, server;
before(async () => {
  const { app } = await import("../index.js");
  server = app.listen(0); base = `http://localhost:${server.address().port}`;
});
after(() => server.close());

function client() {
  let cookie = "";
  const call = async (path, { method = "GET", body, csrf = true } = {}) => {
    const r = await fetch(base + path, { method, headers: { ...(body ? { "Content-Type": "application/json" } : {}), ...(csrf ? { "X-Studyloom": "1" } : {}), ...(cookie ? { Cookie: cookie } : {}) }, body: body && JSON.stringify(body) });
    const set = r.headers.get("set-cookie"); if (set) cookie = set.split(";")[0];
    return { status: r.status, json: await r.json().catch(() => null), headers: r.headers };
  };
  call.cookie = () => cookie;
  return call;
}
const day = "2026-11-12";

test("catalogue is built from the notes", async () => {
  const r = await client()("/api/stats");
  assert.equal(r.status, 200);
  assert.ok(r.json.notes === 19 && r.json.subjects === 4 && r.json.questions === 13, JSON.stringify(r.json));
});

test("the DL subjects come in course order, theory first", async () => {
  const { subjects, notes } = (await client()("/api/catalogue")).json;
  assert.equal(subjects[0].id, "d1-neural-network-foundations");
  assert.equal(subjects[0].track, "theory");
  assert.deepEqual(subjects.map(s => s.num), [...subjects.map(s => s.num)].sort());
  assert.equal(notes.find(n => n.subject_id === "d1-neural-network-foundations").path, "d1-neural-network-foundations/01_foundations.html");
  assert.deepEqual(subjects.map(s => s.track), ["theory", "theory", "pytorch", "architectures"]);   // grows as D2–D5 are written
});

test("sign up, session cookie, me, logout", async () => {
  const c = client();
  const s = await c("/api/auth/signup", { method: "POST", body: { name: "Asha", email: "Asha@Example.com", password: "long-enough-1" } });
  assert.equal(s.status, 201);
  assert.match(s.headers.get("set-cookie"), /HttpOnly/i);
  assert.match(s.headers.get("set-cookie"), /SameSite=Lax/i);
  assert.equal((await c("/api/auth/me")).json.user.email, "asha@example.com");
  await c("/api/auth/logout", { method: "POST" });
  assert.equal((await c("/api/auth/me")).json.user, null);
});

test("validation and duplicate emails", async () => {
  const c = client();
  assert.equal((await c("/api/auth/signup", { method: "POST", body: { name: "B", email: "not-an-email", password: "long-enough-1" } })).status, 400);
  assert.equal((await c("/api/auth/signup", { method: "POST", body: { name: "B", email: "b@example.com", password: "short" } })).json.field, "password");
  await c("/api/auth/signup", { method: "POST", body: { name: "B", email: "b@example.com", password: "long-enough-1" } });
  assert.equal((await client()("/api/auth/signup", { method: "POST", body: { name: "B2", email: "B@example.com", password: "long-enough-1" } })).status, 409);
});

test("wrong password is rejected without saying which part was wrong", async () => {
  const r = await client()("/api/auth/login", { method: "POST", body: { email: "b@example.com", password: "nope-nope" } });
  assert.equal(r.status, 401);
  assert.doesNotMatch(r.json.error, /exist|unknown|no account/i);
});

test("state-changing requests need the X-Studyloom header (CSRF)", async () => {
  const c = client();
  await c("/api/auth/signup", { method: "POST", body: { name: "C", email: "c@example.com", password: "long-enough-1" } });
  const r = await c("/api/progress", { method: "POST", csrf: false, body: { noteId: "d1-neural-network-foundations/02_neuron_and_perceptron", sectionId: "the-biological-neuron", done: true, today: day } });
  assert.equal(r.status, 403);
});

test("progress, streak and per-user isolation", async () => {
  const a = client(), b = client();
  await a("/api/auth/signup", { method: "POST", body: { name: "D", email: "d@example.com", password: "long-enough-1" } });
  await b("/api/auth/signup", { method: "POST", body: { name: "E", email: "e@example.com", password: "long-enough-1" } });
  const p = await a("/api/progress", { method: "POST", body: { noteId: "d1-neural-network-foundations/02_neuron_and_perceptron", sectionId: "the-biological-neuron", done: true, today: day } });
  assert.deepEqual(p.json.done, ["the-biological-neuron"]);
  assert.equal((await a(`/api/dashboard?today=${day}`)).json.streak, 1);
  assert.deepEqual((await b("/api/notes/d1-neural-network-foundations/02_neuron_and_perceptron")).json.me.done, []);
  assert.equal((await a("/api/progress", { method: "POST", body: { noteId: "no/such-note", sectionId: "x", done: true, today: day } })).status, 400);
});

test("my notes: only the owner can edit or delete", async () => {
  const a = client(), b = client();
  await a("/api/auth/signup", { method: "POST", body: { name: "F", email: "f@example.com", password: "long-enough-1" } });
  await b("/api/auth/signup", { method: "POST", body: { name: "G", email: "g@example.com", password: "long-enough-1" } });
  const n = await a("/api/my-notes", { method: "POST", body: { noteId: "d1-neural-network-foundations/02_neuron_and_perceptron", sectionId: "the-biological-neuron", body: "a perceptron is a line: w·x + b = 0" } });
  assert.equal(n.status, 201);
  assert.equal((await b(`/api/my-notes/${n.json.note.id}`, { method: "PATCH", body: { body: "hacked" } })).status, 404);
  await b(`/api/my-notes/${n.json.note.id}`, { method: "DELETE" });
  assert.equal((await a("/api/my-notes")).json.notes[0].body, "a perceptron is a line: w·x + b = 0");
});

test("search is safe against FTS syntax", async () => {
  const c = client();
  for (const q of ['"', "NEAR(", "a OR", "*", "ridge) OR (x"]) assert.equal((await c(`/api/search?q=${encodeURIComponent(q)}`)).status, 200);
  assert.ok((await c("/api/search?q=perceptron")).json.results.length > 0);
});

test("content: notes get the reader kit, private paths don't load", async () => {
  const r = await fetch(base + "/content/d1-neural-network-foundations/01_foundations.html");
  const html = await r.text();
  assert.match(html, /reader-kit\/bridge\.js/);
  assert.match(html, /\.\.\/assets\/notes\.js/);
  assert.equal((await fetch(base + "/content/assets/notes.js")).status, 200);   // what that relative link resolves to
  for (const p of ["/content/..%2fserver%2fdb.js", "/content/..%2fpackage.json", "/content/..%2f..%2fREADME.md", "/content/.git/config", "/content/d1-neural-network-foundations/..%2f..%2fpackage.json"]) assert.equal((await fetch(base + p)).status, 404, p);
});

test("deleting the account needs the password and removes everything", async () => {
  const c = client();
  await c("/api/auth/signup", { method: "POST", body: { name: "H", email: "h@example.com", password: "long-enough-1" } });
  await c("/api/bookmarks", { method: "POST", body: { noteId: "d1-neural-network-foundations/02_neuron_and_perceptron", sectionId: "the-biological-neuron", on: true } });
  assert.equal((await c("/api/account", { method: "DELETE", body: { password: "wrong-one-1" } })).status, 400);
  assert.equal((await c("/api/account", { method: "DELETE", body: { password: "long-enough-1" } })).status, 200);
  assert.equal((await client()("/api/auth/login", { method: "POST", body: { email: "h@example.com", password: "long-enough-1" } })).status, 401);
});

test("exam hall: no question papers yet, and the list is well formed", async () => {
  const c = client();
  await c("/api/auth/signup", { method: "POST", body: { name: "E", email: "exam@example.com", password: "long-enough-1" } });
  const r = await c("/api/papers");
  assert.equal(r.status, 200);
  assert.ok(Array.isArray(r.json.papers) && r.json.papers.every(p => p.source === "notes" || p.source === "pyq"));
});

// the live kernel: real Python from the project's .venv (skipped when there is none)
const { enabled: kernelOn } = await import("../kernels.js");
async function runCell(c, note, cell, code) {
  const r = await fetch(base + "/api/kernel/run", { method: "POST", headers: { "Content-Type": "application/json", "X-Studyloom": "1", Cookie: c.cookie() }, body: JSON.stringify({ note, cell, code }) });
  if (r.status !== 200) return { status: r.status };
  const events = (await r.text()).trim().split("\n").map(l => JSON.parse(l));
  return { status: 200, events, text: events.filter(e => e.t === "text").map(e => e.d).join(""), done: events.at(-1) };
}

test("live kernel: signed-in only, cells share state, errors and restarts", { skip: !kernelOn() && "no .venv" }, async () => {
  const anon = client();
  assert.equal((await anon("/api/kernel/run", { method: "POST", body: { note: "x", cell: 0, code: "1" } })).status, 401);
  const c = client();
  await c("/api/auth/signup", { method: "POST", body: { name: "K", email: "kernel@example.com", password: "long-enough-1" } });
  const note = "d1-neural-network-foundations/01_foundations";
  const a = await runCell(c, note, 0, "x = 6 * 7\nprint(x)");
  assert.equal(a.status, 200); assert.equal(a.text.trim(), "42"); assert.equal(a.done.ok, true);
  const b = await runCell(c, note, 1, "x + 1");                      // the last expression's value, like a notebook
  assert.match(b.text, /43/);
  assert.deepEqual((await c(`/api/kernel/status?note=${note}`)).json.ran.sort(), [0, 1]);
  const e = await runCell(c, note, 2, "1 / 0");
  assert.equal(e.done.ok, false); assert.ok(e.events.some(ev => ev.t === "err" && /ZeroDivisionError/.test(ev.d)));
  const other = await runCell(c, "d1-neural-network-foundations/02_neuron_and_perceptron", 0, "print('x' in dir())");
  assert.equal(other.text.trim(), "False", "each note has its own kernel");
  for (const n of [note, "d1-neural-network-foundations/02_neuron_and_perceptron"]) assert.equal((await c("/api/kernel/restart", { method: "POST", body: { note: n } })).status, 200);
  assert.deepEqual((await c(`/api/kernel/status?note=${note}`)).json, { enabled: true, alive: false, busy: false, ran: [] });
});

// keep last: it exhausts the auth rate limit for everyone after it
test("too many login attempts are rate-limited", async () => {
  const c = client(); let last;
  for (let i = 0; i < 60; i++) last = await c("/api/auth/login", { method: "POST", body: { email: "nobody@example.com", password: "x" } });
  assert.equal(last.status, 429);
  assert.ok(last.headers.get("retry-after"));
});
