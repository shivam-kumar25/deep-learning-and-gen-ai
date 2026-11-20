// Accounts and sessions.
// - passwords: bcrypt (cost 12)
// - sessions: a random 32-byte token in an httpOnly cookie; only its SHA-256 is stored, so a leaked
//   database can't be used to log in. Sliding expiry: 30 days, renewed when less than 15 are left.
// - CSRF: cookies are SameSite=Lax and every state-changing request must carry the X-Studyloom header,
//   which a cross-site form can't set and a cross-site fetch can't send without a CORS preflight we never allow.
import crypto from "node:crypto";
import bcrypt from "bcryptjs";
import { z } from "zod";
import { db } from "./db.js";

const COOKIE = "sl_session";
const DAYS = 30;
const sha = t => crypto.createHash("sha256").update(t).digest("hex");
const PROD = process.env.NODE_ENV === "production";

export const publicUser = u => u && { id: u.id, email: u.email, name: u.name, theme: u.theme, dailyGoal: u.daily_goal, createdAt: u.created_at };

function startSession(res, userId, ua) {
  const token = crypto.randomBytes(32).toString("base64url");
  const expires = new Date(Date.now() + DAYS * 864e5);
  db.prepare("INSERT INTO sessions (id, user_id, expires_at, user_agent) VALUES (?, ?, ?, ?)").run(sha(token), userId, expires.toISOString(), (ua || "").slice(0, 200));
  res.cookie(COOKIE, token, { httpOnly: true, sameSite: "lax", secure: PROD, expires, path: "/" });
}

export function loadUser(req, res, next) {
  const token = req.cookies?.[COOKIE];
  req.user = null;
  if (token) {
    const row = db.prepare(`SELECT s.id sid, s.expires_at, u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.id = ?`).get(sha(token));
    if (row && new Date(row.expires_at) > new Date()) {
      req.user = row; req.sessionId = row.sid;
      if (new Date(row.expires_at) - Date.now() < 15 * 864e5) {
        const expires = new Date(Date.now() + DAYS * 864e5);
        db.prepare("UPDATE sessions SET expires_at = ? WHERE id = ?").run(expires.toISOString(), row.sid);
        res.cookie(COOKIE, token, { httpOnly: true, sameSite: "lax", secure: PROD, expires, path: "/" });
      }
    } else if (row) db.prepare("DELETE FROM sessions WHERE id = ?").run(row.sid);
  }
  next();
}

export function requireUser(req, res, next) {
  if (!req.user) return res.status(401).json({ error: "Please sign in." });
  next();
}

export function csrf(req, res, next) {
  if (["GET", "HEAD", "OPTIONS"].includes(req.method)) return next();
  if (req.get("X-Studyloom") !== "1") return res.status(403).json({ error: "Missing request header." });
  next();
}

// a small fixed-window limiter, per IP + action (enough for one server process)
const hits = new Map();
export const limit = (key, max, windowMs) => (req, res, next) => {
  const k = `${key}:${req.ip}`, now = Date.now();
  const h = hits.get(k) || { n: 0, reset: now + windowMs };
  if (now > h.reset) { h.n = 0; h.reset = now + windowMs; }
  h.n++; hits.set(k, h);
  if (h.n > max * Number(process.env.SL_LIMIT_SCALE || 1)) { res.set("Retry-After", Math.ceil((h.reset - now) / 1000)); return res.status(429).json({ error: "Too many attempts. Please wait a minute and try again." }); }
  next();
};

const Email = z.string().trim().toLowerCase().email("Enter a valid email address.").max(254);
const Password = z.string().min(8, "Use at least 8 characters.").max(200);
const SignUp = z.object({ name: z.string().trim().min(1, "Tell us your name.").max(80), email: Email, password: Password });
const SignIn = z.object({ email: Email, password: z.string().min(1, "Enter your password.").max(200) });
const fail = (res, r) => res.status(400).json({ error: r.error.issues[0].message, field: r.error.issues[0].path[0] });

const DUMMY = bcrypt.hashSync("timing-equaliser", 12);   // compare against this when the email is unknown

export function routes(app) {
  app.post("/api/auth/signup", limit("signup", 10, 60_000), async (req, res) => {
    const r = SignUp.safeParse(req.body); if (!r.success) return fail(res, r);
    const { name, email, password } = r.data;
    if (db.prepare("SELECT 1 FROM users WHERE email = ?").get(email)) return res.status(409).json({ error: "An account with this email already exists. Sign in instead.", field: "email" });
    const hash = await bcrypt.hash(password, 12);
    const { lastInsertRowid } = db.prepare("INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)").run(email, name, hash);
    startSession(res, lastInsertRowid, req.get("user-agent"));
    res.status(201).json({ user: publicUser(db.prepare("SELECT * FROM users WHERE id = ?").get(lastInsertRowid)) });
  });

  app.post("/api/auth/login", limit("login", 10, 60_000), async (req, res) => {
    const r = SignIn.safeParse(req.body); if (!r.success) return fail(res, r);
    const u = db.prepare("SELECT * FROM users WHERE email = ?").get(r.data.email);
    const ok = await bcrypt.compare(r.data.password, u?.password_hash || DUMMY);
    if (!u || !ok) return res.status(401).json({ error: "That email and password don't match." });
    startSession(res, u.id, req.get("user-agent"));
    res.json({ user: publicUser(u) });
  });

  app.post("/api/auth/logout", (req, res) => {
    if (req.sessionId) db.prepare("DELETE FROM sessions WHERE id = ?").run(req.sessionId);
    res.clearCookie(COOKIE, { path: "/" });
    res.json({ ok: true });
  });

  app.get("/api/auth/me", (req, res) => res.json({ user: publicUser(req.user) }));

  const Prefs = z.object({ name: z.string().trim().min(1).max(80).optional(), theme: z.enum(["system", "light", "dark"]).optional(), dailyGoal: z.number().int().min(5).max(600).optional() });
  app.patch("/api/account", requireUser, (req, res) => {
    const r = Prefs.safeParse(req.body); if (!r.success) return fail(res, r);
    const u = { ...publicUser(req.user), ...r.data };
    db.prepare("UPDATE users SET name = ?, theme = ?, daily_goal = ? WHERE id = ?").run(u.name, u.theme, u.dailyGoal, req.user.id);
    res.json({ user: publicUser(db.prepare("SELECT * FROM users WHERE id = ?").get(req.user.id)) });
  });

  const PwChange = z.object({ current: z.string().min(1).max(200), next: Password });
  app.post("/api/account/password", requireUser, limit("pw", 10, 60_000), async (req, res) => {
    const r = PwChange.safeParse(req.body); if (!r.success) return fail(res, r);
    if (!(await bcrypt.compare(r.data.current, req.user.password_hash))) return res.status(400).json({ error: "Your current password is not right.", field: "current" });
    db.prepare("UPDATE users SET password_hash = ? WHERE id = ?").run(await bcrypt.hash(r.data.next, 12), req.user.id);
    db.prepare("DELETE FROM sessions WHERE user_id = ? AND id != ?").run(req.user.id, req.sessionId);   // sign out other devices
    res.json({ ok: true });
  });

  app.delete("/api/account", requireUser, limit("del", 5, 60_000), async (req, res) => {
    if (!(await bcrypt.compare(String(req.body?.password || ""), req.user.password_hash))) return res.status(400).json({ error: "Your password is not right.", field: "password" });
    db.prepare("DELETE FROM users WHERE id = ?").run(req.user.id);       // cascades to every table
    res.clearCookie(COOKIE, { path: "/" });
    res.json({ ok: true });
  });
}
