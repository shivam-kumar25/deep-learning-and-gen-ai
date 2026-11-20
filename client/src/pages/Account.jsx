import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Empty, Icon } from "../components.jsx";
import { api, hueStyle, timeAgo, useAuth, useFetch, useToast } from "../lib.jsx";

export function Bookmarks() {
  const { data, setData } = useFetch("/api/bookmarks");
  const remove = async b => { await api("/api/bookmarks", { method: "POST", body: { noteId: b.note_id, sectionId: b.section_id, on: false } }); setData(d => ({ bookmarks: d.bookmarks.filter(x => x !== b) })); };
  return <div className="page">
    <div className="page-head"><div><span className="eyebrow">Bookmarks</span><h1 className="display" style={{ marginTop: 8 }}>Sections to come back to.</h1><p>Bookmark any section from the reader's top bar.</p></div></div>
    {!data ? <div className="skeleton" style={{ height: 200 }} /> : data.bookmarks.length ? <div className="card band"><div className="list">
      {data.bookmarks.map(b => <div key={b.note_id + b.section_id} className="list-item" style={hueStyle(b.hue)}>
        <span className="subject-icon sm"><Icon name="Bookmark" size={16} /></span>
        <Link className="grow" to={`/app/learn/${b.note_id}${b.section_id ? "#" + b.section_id : ""}`} style={{ color: "var(--ink)", textDecoration: "none" }}><span className="li-t">{b.section_title || b.title}</span><span className="li-s">{b.subject_title} · {b.title} · saved {timeAgo(b.created_at)}</span></Link>
        <button className="btn ghost icon" aria-label="Remove bookmark" onClick={() => remove(b)}><Icon name="X" size={17} /></button>
      </div>)}</div></div> : <div className="card"><Empty icon="Bookmark" title="No bookmarks yet" action={<Link className="btn primary" to="/app/library">Find something to read</Link>}>Tap the bookmark icon while reading to save a section here.</Empty></div>}
  </div>;
}

export function MyNotes() {
  const { data } = useFetch("/api/my-notes");
  const [q, setQ] = useState("");
  const list = (data?.notes || []).filter(n => !q || (n.body + n.title + (n.section_title || "")).toLowerCase().includes(q.toLowerCase()));
  return <div className="page">
    <div className="page-head"><div><span className="eyebrow">My notes</span><h1 className="display" style={{ marginTop: 8 }}>Everything you wrote down.</h1><p>Each note stays attached to the section it's about.</p></div>
      {!!data?.notes.length && <input className="input" style={{ maxWidth: 300 }} placeholder="Filter your notes…" value={q} onChange={e => setQ(e.target.value)} aria-label="Filter notes" />}</div>
    {!data ? <div className="skeleton" style={{ height: 200 }} /> : list.length ? <div className="grid-subjects">
      {list.map(n => <Link key={n.id} to={`/app/learn/${n.note_id}${n.section_id ? "#" + n.section_id : ""}`} className="subject-card" style={hueStyle(n.hue)}>
        <span className="chip tint" style={{ justifySelf: "start" }}>{n.subject_title}</span>
        <p style={{ whiteSpace: "pre-wrap", color: "var(--ink)", display: "-webkit-box", WebkitLineClamp: 6, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{n.body}</p>
        <div className="meta"><span>{n.section_title || n.title}</span><span>{timeAgo(n.updated_at)}</span></div>
      </Link>)}</div> : <div className="card"><Empty icon="NotebookPen" title={q ? "No notes match" : "No notes yet"}>{q ? "Try another word." : "Open the Notebook panel while reading (or press N) to write one."}</Empty></div>}
  </div>;
}

export function Settings() {
  const { user, setUser, pref, setPref, logout } = useAuth();
  const toast = useToast(), nav = useNavigate();
  const [name, setName] = useState(user.name), [goal, setGoal] = useState(user.dailyGoal);
  const [pw, setPw] = useState({ current: "", next: "" }), [pwMsg, setPwMsg] = useState(null);
  const [del, setDel] = useState(""), [delErr, setDelErr] = useState(null);
  const save = async e => { e.preventDefault(); try { const r = await api("/api/account", { method: "PATCH", body: { name, dailyGoal: Number(goal) } }); setUser(r.user); toast("Saved", "✓"); } catch (x) { toast(x.message, "⚠️"); } };
  const changePw = async e => { e.preventDefault(); try { await api("/api/account/password", { method: "POST", body: pw }); setPw({ current: "", next: "" }); setPwMsg({ ok: true, t: "Password changed. Other devices were signed out." }); } catch (x) { setPwMsg({ ok: false, t: x.message }); } };
  const remove = async e => { e.preventDefault(); try { await api("/api/account", { method: "DELETE", body: { password: del } }); await logout(); nav("/"); } catch (x) { setDelErr(x.message); } };
  return <div className="page" style={{ maxWidth: 820 }}>
    <div className="page-head"><div><span className="eyebrow">Settings</span><h1 className="display" style={{ marginTop: 8 }}>Your account</h1></div></div>
    <form className="card band stack" style={{ "--gap": "16px" }} onSubmit={save}>
      <h2 className="display" style={{ fontSize: 21 }}>Profile and goal</h2>
      <div className="field"><label htmlFor="n">Name</label><input id="n" className="input" value={name} onChange={e => setName(e.target.value)} maxLength={80} /></div>
      <div className="field"><label>Email</label><input className="input" value={user.email} disabled /></div>
      <div className="field"><label htmlFor="g">Daily goal: <b>{goal} minutes</b></label><input id="g" type="range" min="5" max="180" step="5" value={goal} onChange={e => setGoal(e.target.value)} style={{ accentColor: "var(--brand)" }} /></div>
      <button className="btn primary" style={{ justifySelf: "start" }}>Save changes</button>
    </form>
    <section className="card band stack" style={{ "--gap": "14px", marginTop: 16 }}>
      <h2 className="display" style={{ fontSize: 21 }}>Appearance</h2>
      <div className="tabs" style={{ maxWidth: 420 }}>{[["system", "Match my device"], ["light", "Paper"], ["dark", "Ink"]].map(([k, l]) => <button key={k} className={pref === k ? "on" : ""} onClick={() => setPref(k)}>{l}</button>)}</div>
    </section>
    <form className="card band stack" style={{ "--gap": "14px", marginTop: 16 }} onSubmit={changePw}>
      <h2 className="display" style={{ fontSize: 21 }}>Password</h2>
      {pwMsg && <div className={"alert" + (pwMsg.ok ? " ok" : "")}>{pwMsg.t}</div>}
      <div className="field"><label htmlFor="c">Current password</label><input id="c" type="password" autoComplete="current-password" className="input" value={pw.current} onChange={e => setPw({ ...pw, current: e.target.value })} /></div>
      <div className="field"><label htmlFor="nw">New password (8+ characters)</label><input id="nw" type="password" autoComplete="new-password" className="input" value={pw.next} onChange={e => setPw({ ...pw, next: e.target.value })} /></div>
      <button className="btn" style={{ justifySelf: "start" }} disabled={!pw.current || pw.next.length < 8}>Change password</button>
    </form>
    <section className="card band stack" style={{ "--gap": "14px", marginTop: 16 }}>
      <h2 className="display" style={{ fontSize: 21 }}>Your data</h2>
      <p className="muted">Download everything we store about you: your account details, progress, study time, bookmarks and notes, as JSON.</p>
      <a className="btn" href="/api/account/export" style={{ justifySelf: "start" }}><Icon name="Download" size={17} />Download my data</a>
    </section>
    <form className="card band stack" style={{ "--gap": "14px", marginTop: 16, borderColor: "color-mix(in srgb, var(--bad) 35%, var(--line))" }} onSubmit={remove}>
      <h2 className="display" style={{ fontSize: 21, color: "var(--bad)" }}>Delete account</h2>
      <p className="muted">This permanently deletes your account and everything in it. It can't be undone.</p>
      {delErr && <div className="alert">{delErr}</div>}
      <div className="field"><label htmlFor="d">Type your password to confirm</label><input id="d" type="password" autoComplete="current-password" className="input" value={del} onChange={e => setDel(e.target.value)} /></div>
      <button className="btn danger" style={{ justifySelf: "start" }} disabled={!del}>Delete my account</button>
    </form>
  </div>;
}

export function NotFound() {
  return <div className="page"><Empty icon="Compass" title="This page wandered off" action={<Link className="btn primary" to="/">Back home</Link>}>The link may be old, or mistyped.</Empty></div>;
}
