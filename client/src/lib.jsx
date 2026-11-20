import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

/* ---------------------------------------------------------------- API */
export class ApiError extends Error { constructor(msg, status, field) { super(msg); this.status = status; this.field = field; } }
export async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method, credentials: "same-origin",
    headers: { "X-Studyloom": "1", ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new ApiError(data?.error || `Request failed (${res.status})`, res.status, data?.field);
  return data;
}
export const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };

export function useFetch(path, deps = []) {
  const [state, set] = useState({ data: null, error: null, loading: true });
  const reload = useCallback(() => {
    if (!path) return;
    set(s => ({ ...s, loading: true }));
    api(path).then(data => set({ data, error: null, loading: false }), error => set({ data: null, error, loading: false }));
  }, [path]);  // eslint-disable-line
  useEffect(reload, [reload, ...deps]);
  return { ...state, reload, setData: d => set(s => ({ ...s, data: typeof d === "function" ? d(s.data) : d })) };
}

/* ---------------------------------------------------------------- auth + theme */
const Auth = createContext(null);
export const useAuth = () => useContext(Auth);
const media = () => matchMedia("(prefers-color-scheme: dark)");
export function resolveTheme(pref) { return pref === "dark" || (pref === "system" && media().matches) ? "dark" : "light"; }

export function AuthProvider({ children }) {
  const [user, setUser] = useState(undefined);              // undefined = still loading
  const [pref, setPref] = useState(() => localStorage.getItem("sl-theme") || "system");
  const [theme, setTheme] = useState(() => resolveTheme(localStorage.getItem("sl-theme") || "system"));
  useEffect(() => { api("/api/auth/me").then(r => setUser(r.user), () => setUser(null)); }, []);
  useEffect(() => { if (user?.theme) setPref(user.theme); }, [user?.theme]);
  useEffect(() => {
    const apply = () => { const t = resolveTheme(pref); setTheme(t); document.documentElement.dataset.theme = t; };
    apply(); localStorage.setItem("sl-theme", pref);
    const m = media(); m.addEventListener("change", apply); return () => m.removeEventListener("change", apply);
  }, [pref]);
  const choose = useCallback(async p => { setPref(p); if (user) { try { const r = await api("/api/account", { method: "PATCH", body: { theme: p } }); setUser(r.user); } catch {} } }, [user]);
  const value = useMemo(() => ({ user, setUser, theme, pref, setPref: choose,
    toggleTheme: () => choose(theme === "dark" ? "light" : "dark"),
    logout: async () => { await api("/api/auth/logout", { method: "POST" }).catch(() => {}); setUser(null); } }), [user, theme, pref, choose]);
  return <Auth.Provider value={value}>{children}</Auth.Provider>;
}

/* ---------------------------------------------------------------- toasts */
const Toasts = createContext(() => {});
export const useToast = () => useContext(Toasts);
export function ToastProvider({ children }) {
  const [items, setItems] = useState([]);
  const push = useCallback((text, icon = "") => {
    const id = Math.random();
    setItems(x => [...x, { id, text, icon }]);
    setTimeout(() => setItems(x => x.filter(t => t.id !== id)), 2600);
  }, []);
  return <Toasts.Provider value={push}>{children}
    <div className="toast-wrap" role="status" aria-live="polite">{items.map(t => <div className="toast" key={t.id}>{t.icon}<span>{t.text}</span></div>)}</div>
  </Toasts.Provider>;
}

/* ---------------------------------------------------------------- small helpers */
export const plural = (n, w, p = w + "s") => `${n.toLocaleString()} ${n === 1 ? w : p}`;
export const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);
export function timeAgo(iso) {
  const t = new Date(iso.replace(" ", "T") + (iso.endsWith("Z") ? "" : "Z")).getTime();
  const s = (Date.now() - t) / 1000, rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (s < 60) return rtf.format(-Math.round(s), "second");
  if (s < 3600) return rtf.format(-Math.round(s / 60), "minute");
  if (s < 86400) return rtf.format(-Math.round(s / 3600), "hour");
  return rtf.format(-Math.round(s / 86400), "day");
}
export const minutes = s => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : `${Math.round(s / 60)} min`);
export function useInterval(fn, ms) { const r = useRef(fn); r.current = fn; useEffect(() => { if (ms == null) return; const t = setInterval(() => r.current(), ms); return () => clearInterval(t); }, [ms]); }
export const hueStyle = h => ({ "--h": h });
/* Maths in question HTML: the notes mark it as <span class="tex">…</span> / <div class="tex-d">…</div>.
   KaTeX is the copy vendored with the notes, loaded once on first use. */
const KATEX = "/content/assets/vendor/katex/";
let katexLoad = null;
function loadKatex() {
  katexLoad = katexLoad || new Promise((res, rej) => {
    if (window.katex) return res();
    const css = document.createElement("link"); css.rel = "stylesheet"; css.href = KATEX + "katex.min.css"; document.head.append(css);
    const s = document.createElement("script"); s.src = KATEX + "katex.min.js"; s.onload = res; s.onerror = rej; document.head.append(s);
  });
  return katexLoad;
}
export function renderMath(root) {
  if (!root) return;
  const items = [...root.querySelectorAll(".tex:not([data-r]), .tex-d:not([data-r])")];
  if (!items.length) return;
  loadKatex().then(() => items.forEach(el => {
    if (el.dataset.r) return;
    el.dataset.r = "1";
    try { window.katex.render(el.textContent, el, { displayMode: el.classList.contains("tex-d"), throwOnError: false, strict: "ignore", trust: false }); } catch { /* leave the source visible */ }
  })).catch(() => {});
}
/** HTML from the server (question prompts, options, explanations), with its maths typeset */
export function MathHtml({ html, className, as: Tag = "div" }) {
  const ref = useRef(null);
  useEffect(() => { renderMath(ref.current); }, [html]);
  return <Tag ref={ref} className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** search snippets come back with \u0002 … \u0003 around matches: render them as <mark>, everything else as text */
export function Snippet({ text }) {
  return <>{String(text || "").split(/(\u0002[^\u0003]*\u0003)/).map((p, i) => p.startsWith("\u0002") ? <mark key={i}>{p.slice(1, -1)}</mark> : p)}</>;
}
