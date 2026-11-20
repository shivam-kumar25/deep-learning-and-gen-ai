import express from "express";
import cookieParser from "cookie-parser";
import fs from "node:fs";
import path from "node:path";
import { db, APP_ROOT, contentFile } from "./db.js";
import { buildCatalogue } from "./indexer.js";
import * as auth from "./auth.js";
import * as api from "./api.js";
import * as papers from "./papers.js";
import * as kernels from "./kernels.js";

if (!db.prepare("SELECT 1 FROM notes LIMIT 1").get() || process.env.REINDEX) buildCatalogue();
if (!db.prepare("SELECT 1 FROM papers LIMIT 1").get() || process.env.REINDEX) papers.buildPapers();

export const app = express();
app.disable("x-powered-by");
app.set("trust proxy", process.env.TRUST_PROXY ? 1 : false);
app.use((req, res, next) => {
  res.set({ "X-Content-Type-Options": "nosniff", "Referrer-Policy": "strict-origin-when-cross-origin", "Cross-Origin-Opener-Policy": "same-origin" });
  next();
});
app.use(express.json({ limit: "64kb" }));
app.use(cookieParser());
app.use(auth.loadUser);
app.use("/api", auth.csrf);
auth.routes(app);
api.routes(app);
papers.routes(app);
kernels.routes(app);
app.use("/api", (req, res) => res.status(404).json({ error: "Not found." }));
app.use("/api", (err, req, res, next) => { console.error(err); res.status(err.status || 500).json({ error: err.expose ? err.message : "Something went wrong on our side." }); });

// ------------------------------------------------------------------ the notes, served with the reader theme
const READER = path.join(APP_ROOT, "server", "reader");
app.use("/reader-kit", express.static(READER, { maxAge: process.env.NODE_ENV === "production" ? "1h" : 0 }));
for (const f of ["inter", "fraunces", "jetbrains-mono"])
  app.use("/reader-kit/fonts", express.static(path.join(APP_ROOT, "node_modules", "@fontsource-variable", f, "files"), { maxAge: "30d" }));
const ALLOWED = /\.(html|css|js|mjs|json|png|jpe?g|gif|svg|webp|ico|txt|py|csv|data|gz|yaml|yml|md|woff2?)$/i;   // .data/.gz: the datasets the notes serve to Pyodide
const INJECT_HEAD = `<link rel="stylesheet" href="/reader-kit/reader.css"><script src="/reader-kit/bridge.js"></script>`;
app.get("/content/*path", (req, res, next) => {
  const rel = [].concat(req.params.path).join("/");
  const file = contentFile(rel);
  const inside = !!file;
  const hidden = rel.split("/").some(p => p.startsWith(".")) || rel.includes("node_modules");
  if (!inside || hidden || !ALLOWED.test(file) || !fs.existsSync(file) || !fs.statSync(file).isFile()) return next();
  if (!/\.html$/i.test(file)) return res.sendFile(file);
  let html = fs.readFileSync(file, "utf8");
  html = /<head[^>]*>/i.test(html) ? html.replace(/<head[^>]*>/i, m => m + INJECT_HEAD) : INJECT_HEAD + html;
  res.type("html").set("Cache-Control", "no-cache").send(html);
});

// ------------------------------------------------------------------ the React app (production build)
const DIST = path.join(APP_ROOT, "dist");
if (fs.existsSync(DIST)) {
  app.use(express.static(DIST, { index: false, maxAge: "1h" }));
  app.get(/^\/(?!api|content|reader-kit).*/, (req, res) => res.sendFile(path.join(DIST, "index.html")));
}

const PORT = Number(process.env.PORT || 4300);
if (process.env.NODE_ENV !== "test") app.listen(PORT, () => console.log(`Studyloom API on http://localhost:${PORT}${fs.existsSync(DIST) ? "" : "  (dev: open the Vite URL, http://localhost:5173)"}`));
