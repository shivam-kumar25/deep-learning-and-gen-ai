import { useEffect, useMemo, useRef, useState } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import * as I from "./icons.js";
import { api, hueStyle, Snippet, useAuth } from "./lib.jsx";

export function Logo({ to = "/" }) {
  return <Link to={to} className="logo" aria-label="Studyloom home">
    <svg viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="var(--brand)" />
      <path d="M8 11.5c4 0 4 4 8 4s4-4 8-4M8 16.5c4 0 4 4 8 4s4-4 8-4" stroke="#fff" strokeWidth="2.2" fill="none" strokeLinecap="round" />
      <path d="M8 21.5c4 0 4 4 8 4s4-4 8-4" stroke="var(--accent)" strokeWidth="2.2" fill="none" strokeLinecap="round" opacity=".95" />
    </svg>
    <span>Studyloom</span>
  </Link>;
}

export function Icon({ name, size = 20, ...p }) {
  const C = I[name] || I.BookOpen;
  return <C size={size} strokeWidth={1.8} aria-hidden="true" {...p} />;
}

export function SubjectIcon({ subject, small }) {
  return <span className={"subject-icon" + (small ? " sm" : "")} style={hueStyle(subject.hue)}><Icon name={subject.icon} size={small ? 17 : 22} /></span>;
}

export function Ring({ value, size = 64, stroke = 7, hue, label }) {
  const r = (size - stroke) / 2, c = 2 * Math.PI * r, v = Math.max(0, Math.min(100, value));
  return <div className="ring" style={{ width: size, height: size, ...(hue != null ? hueStyle(hue) : {}) }} role="img" aria-label={label || `${v}% complete`}>
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--paper-2)" strokeWidth={stroke} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--tint-strong)" strokeWidth={stroke} strokeLinecap="round"
              strokeDasharray={c} strokeDashoffset={c * (1 - v / 100)} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: "stroke-dashoffset .8s var(--ease)" }} />
    </svg>
    <span>{v}%</span>
  </div>;
}

/** the last 26 weeks of study, one square per day */
export function Heatmap({ days, today }) {
  const map = useMemo(() => Object.fromEntries(days.map(d => [d.day, d])), [days]);
  const cells = [];
  const end = new Date(today + "T12:00:00Z");
  const start = new Date(end); start.setUTCDate(start.getUTCDate() - (7 * 25 + end.getUTCDay()));
  for (let d = new Date(start); d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
    const k = d.toISOString().slice(0, 10), x = map[k];
    const m = x ? x.seconds / 60 + x.sections * 4 : 0;
    const lvl = m === 0 ? 0 : m < 10 ? 1 : m < 25 ? 2 : m < 50 ? 3 : 4;
    cells.push(<i key={k} className={"l" + lvl} title={`${new Date(k + "T12:00:00Z").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}: ${x ? `${Math.round(x.seconds / 60)} min, ${x.sections} sections` : "no study"}`} />);
  }
  return <div className="heatmap" aria-label="Study activity over the last six months">{cells}</div>;
}

/* ---------------------------------------------------------------- search (Ctrl/⌘ K) */
export function SearchPalette({ open, onClose }) {
  const [q, setQ] = useState(""), [res, setRes] = useState([]), [sel, setSel] = useState(0);
  const nav = useNavigate(), inp = useRef(null);
  useEffect(() => { if (open) { setQ(""); setRes([]); setTimeout(() => inp.current?.focus(), 20); } }, [open]);
  useEffect(() => {
    if (q.trim().length < 2) { setRes([]); return; }
    const t = setTimeout(() => api(`/api/search?q=${encodeURIComponent(q)}`).then(r => { setRes(r.results); setSel(0); }), 140);
    return () => clearTimeout(t);
  }, [q]);
  if (!open) return null;
  const go = r => { onClose(); nav(`/app/learn/${r.note_id}${r.section_id ? "#" + r.section_id : ""}`); };
  return <div className="palette-bg" onMouseDown={onClose}>
    <div className="palette" role="dialog" aria-modal="true" aria-label="Search the notes" onMouseDown={e => e.stopPropagation()}>
      <div className="palette-in"><Icon name="Search" />
        <input ref={inp} value={q} onChange={e => setQ(e.target.value)} placeholder="Search every note: “cross_val_score”, “ridge”, “silhouette”…"
          onKeyDown={e => {
            if (e.key === "Escape") onClose();
            if (e.key === "ArrowDown") { e.preventDefault(); setSel(s => Math.min(s + 1, res.length - 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setSel(s => Math.max(s - 1, 0)); }
            if (e.key === "Enter" && res[sel]) go(res[sel]);
          }} aria-label="Search" />
        <kbd>Esc</kbd>
      </div>
      <div className="palette-list">
        {q.trim().length < 2 && <p className="palette-hint">Search every note and section at once. Use ↑ ↓ and Enter.</p>}
        {q.trim().length >= 2 && !res.length && <p className="palette-hint">Nothing found for “{q}”.</p>}
        {res.map((r, i) => <button key={r.note_id + r.section_id} className={"palette-item" + (i === sel ? " on" : "")} onMouseEnter={() => setSel(i)} onClick={() => go(r)} style={hueStyle(r.hue)}>
          <span className="dot" />
          <span className="grow">
            <span className="pi-t">{r.section_title || r.note_title}</span>
            <span className="pi-s">{r.subject_title} · {r.note_title}</span>
            <span className="pi-x"><Snippet text={r.snippet} /></span>
          </span>
          <Icon name="CornerDownLeft" size={16} />
        </button>)}
      </div>
    </div>
  </div>;
}

/* ---------------------------------------------------------------- the signed-in shell */
const NAV = [
  ["/app", "Today", "Sun", true],
  ["/app/library", "Library", "LibraryBig"],
  ["/app/exams", "Exam hall", "ClipboardCheck"],
  ["/app/bookmarks", "Bookmarks", "Bookmark"],
  ["/app/my-notes", "My notes", "NotebookPen"],
  ["/app/settings", "Settings", "Settings2"],
];
export function AppShell() {
  const { user, theme, toggleTheme, logout } = useAuth();
  const [search, setSearch] = useState(false), [menu, setMenu] = useState(false);
  const nav = useNavigate(), loc = useLocation();
  const reader = loc.pathname.startsWith("/app/learn/") || loc.pathname.startsWith("/app/attempts/");
  useEffect(() => {
    const k = e => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setSearch(s => !s); } };
    addEventListener("keydown", k); window.__slOpenSearch = () => setSearch(true);
    return () => removeEventListener("keydown", k);
  }, []);
  useEffect(() => setMenu(false), [loc.pathname]);
  return <div className={"shell" + (reader ? " reading" : "")}>
    <aside className={"side" + (menu ? " open" : "")}>
      <div className="side-top"><Logo to="/app" /></div>
      <button className="search-btn" onClick={() => setSearch(true)}><Icon name="Search" size={17} /><span>Search notes</span><kbd>Ctrl K</kbd></button>
      <nav className="side-nav" aria-label="Main">
        {NAV.map(([to, label, icon, end]) => <NavLink key={to} to={to} end={end}><Icon name={icon} size={19} />{label}</NavLink>)}
      </nav>
      <div className="side-foot">
        <button className="btn ghost sm" onClick={toggleTheme} aria-label="Toggle light and dark"><Icon name={theme === "dark" ? "SunMedium" : "MoonStar"} size={17} />{theme === "dark" ? "Light" : "Dark"}</button>
        <div className="me">
          <span className="avatar" aria-hidden="true">{(user?.name || "?").trim()[0]?.toUpperCase()}</span>
          <span className="grow"><b>{user?.name}</b><small>{user?.email}</small></span>
          <button className="btn ghost icon" title="Sign out" aria-label="Sign out" onClick={async () => { await logout(); nav("/"); }}><Icon name="LogOut" size={17} /></button>
        </div>
      </div>
    </aside>
    <div className="main">
      <header className="mobile-top"><button className="btn ghost icon" aria-label="Menu" onClick={() => setMenu(m => !m)}><Icon name="Menu" /></button><Logo to="/app" /><button className="btn ghost icon" aria-label="Search" onClick={() => setSearch(true)}><Icon name="Search" /></button></header>
      <Outlet />
    </div>
    {menu && <div className="scrim" onClick={() => setMenu(false)} />}
    <SearchPalette open={search} onClose={() => setSearch(false)} />
  </div>;
}

export function Empty({ icon, title, children, action }) {
  return <div className="empty"><span className="empty-ic"><Icon name={icon} size={26} /></span><h3 className="display">{title}</h3><p className="muted">{children}</p>{action}</div>;
}
