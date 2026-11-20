// Live Python kernels for the notes: real CPython with PyTorch (and the GPU) on this machine.
// One kernel per (signed-in user, note), so a note's cells share state like a notebook and two notes never collide.
// The browser sends a cell; the answer streams back as NDJSON (one output event per line) while the cell runs.
//
//   POST /api/kernel/run      {note, cell, code}   → streamed events, then {done, ok}
//   GET  /api/kernel/status?note=…                 → {enabled, alive, busy, ran: [cell indexes run in this kernel]}
//   POST /api/kernel/restart  {note}               → kills the kernel (also how a running cell is stopped)
//
// Safety: only for signed-in users, only from this machine (or a private network when KERNEL_ALLOW_LAN=1),
// a cap on live kernels, idle kernels shut down. Off when KERNEL=off.
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { z } from "zod";
import { APP_ROOT } from "./db.js";
import { requireUser } from "./auth.js";

const PY = process.env.KERNEL_PYTHON || path.join(APP_ROOT, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const WORKER = path.join(APP_ROOT, "tools", "kernel", "worker.py");
const WORKDIR = path.join(APP_ROOT, "workspace");          // cells run here: data/ lives under it
const MAX = Number(process.env.KERNEL_MAX || 4);
const IDLE_MS = Number(process.env.KERNEL_IDLE_MIN || 30) * 60_000;
export const enabled = () => process.env.KERNEL !== "off" && fs.existsSync(PY);

const kernels = new Map();   // key → kernel

function allowedAddress(req) {
  const a = (req.socket.remoteAddress || "").replace(/^::ffff:/, "");
  if (a === "127.0.0.1" || a === "::1") return true;
  return process.env.KERNEL_ALLOW_LAN === "1" && /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(a);
}

function start(key) {
  fs.mkdirSync(WORKDIR, { recursive: true });
  const proc = spawn(PY, ["-u", WORKER], { cwd: WORKDIR, windowsHide: true,
    env: { ...process.env, PYTHONIOENCODING: "utf-8", MPLBACKEND: "agg", PYTHONUNBUFFERED: "1" } });
  const k = { key, proc, buf: "", waiting: new Map(), ran: new Set(), busy: false, seq: 0, last: Date.now(), ready: null, dead: false, queue: Promise.resolve() };
  k.ready = new Promise((resolve, reject) => { k.onReady = resolve; k.onFail = reject; });
  k.ready.catch(() => {});                                 // a kernel killed before it was ready must not crash the server
  proc.stdout.setEncoding("utf8");
  proc.stdout.on("data", chunk => {
    k.buf += chunk;
    let i;
    while ((i = k.buf.indexOf("\n")) >= 0) {
      const line = k.buf.slice(0, i); k.buf = k.buf.slice(i + 1);
      let ev; try { ev = JSON.parse(line); } catch { continue; }
      if (ev.ready) { k.onReady(ev); continue; }
      k.waiting.get(ev.id)?.(ev);
    }
  });
  proc.stderr.on("data", () => {});                       // library chatter (CUDA, warnings outside cells): not shown
  proc.on("exit", () => {
    k.dead = true; k.onFail(new Error("kernel exited"));
    for (const cb of k.waiting.values()) cb({ done: true, ok: false, t: "err", d: "The Python kernel stopped (restarted, stopped, or out of memory). Run the cell again to start a fresh one.\n" });
    if (kernels.get(key) === k) kernels.delete(key);
  });
  kernels.set(key, k);
  return k;
}

function kill(k) { if (k && !k.dead) { k.dead = true; try { k.proc.kill(); } catch {} } }

function get(key) {
  let k = kernels.get(key);
  if (k && !k.dead) { k.last = Date.now(); return k; }
  if (kernels.size >= MAX) {                               // evict the longest-idle kernel that is not busy
    const idle = [...kernels.values()].filter(x => !x.busy).sort((a, b) => a.last - b.last)[0];
    if (!idle) return null;
    kill(idle); kernels.delete(idle.key);
  }
  return start(key);
}

setInterval(() => { for (const k of kernels.values()) if (!k.busy && Date.now() - k.last > IDLE_MS) { kill(k); kernels.delete(k.key); } }, 60_000).unref();
for (const sig of ["SIGINT", "SIGTERM", "exit"]) process.on(sig, () => { for (const k of kernels.values()) kill(k); if (sig !== "exit") process.exit(0); });

const keyOf = (req, note) => `${req.user.id}:${note}`;
const Note = z.string().min(1).max(200).regex(/^[\w./-]+$/);

export function routes(app) {
  const guard = (req, res, next) => {
    if (!enabled()) return res.status(503).json({ error: "Live Python is turned off on this server." });
    if (!allowedAddress(req)) return res.status(403).json({ error: "Live Python only runs for this computer." });
    next();
  };

  app.get("/api/kernel/status", requireUser, (req, res) => {
    const note = Note.safeParse(req.query.note);
    if (!note.success) return res.status(400).json({ error: "Bad note." });
    const k = kernels.get(keyOf(req, note.data));
    res.json({ enabled: enabled() && allowedAddress(req), alive: !!k && !k.dead, busy: !!k?.busy, ran: k ? [...k.ran] : [] });
  });

  app.post("/api/kernel/restart", requireUser, guard, (req, res) => {
    const note = Note.safeParse(req.body?.note);
    if (!note.success) return res.status(400).json({ error: "Bad note." });
    const key = keyOf(req, note.data);
    kill(kernels.get(key)); kernels.delete(key);
    res.json({ ok: true });
  });

  const Run = z.object({ note: Note, cell: z.number().int().min(-1).max(10_000), code: z.string().max(60_000) });
  app.post("/api/kernel/run", requireUser, guard, async (req, res) => {
    const body = Run.safeParse(req.body);
    if (!body.success) return res.status(400).json({ error: "Bad request." });
    const { note, cell, code } = body.data;
    const k = get(keyOf(req, note));
    if (!k) return res.status(429).json({ error: `All ${MAX} live kernels are busy. Wait for a cell to finish, or restart one.` });
    res.status(200).set({ "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no" });
    res.flushHeaders?.();
    const send = ev => { if (!res.writableEnded) res.write(JSON.stringify(ev) + "\n"); };
    // one cell at a time per kernel, in arrival order
    k.queue = k.queue.then(() => new Promise(async done => {
      if (k.dead) { send({ done: true, ok: false, t: "err", d: "The kernel stopped. Run the cell again.\n" }); res.end(); return done(); }
      try { const info = await k.ready; if (!k.announced) { k.announced = true; send({ t: "info", d: `Python ${info.python} kernel started` }); } }
      catch { send({ done: true, ok: false, t: "err", d: "Could not start Python. Is the project's .venv set up?\n" }); res.end(); return done(); }
      const id = ++k.seq;
      k.busy = true; k.last = Date.now();
      k.waiting.set(id, ev => {
        if (ev.done) {
          k.waiting.delete(id); k.busy = false; k.last = Date.now();
          if (ev.d) send({ t: ev.t, d: ev.d });
          if (ev.ok && cell >= 0) k.ran.add(cell);
          send({ done: true, ok: !!ev.ok }); res.end(); done();
        } else send({ t: ev.t, d: ev.d });
      });
      // the browser went away mid-cell: keep the kernel (the cell finishes), just stop streaming
      res.on("close", () => {});
      k.proc.stdin.write(JSON.stringify({ id, code }) + "\n");
    }));
  });
}
