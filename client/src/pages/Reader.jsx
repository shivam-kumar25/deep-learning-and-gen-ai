import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { Empty, Icon } from "../components.jsx";
import { api, hueStyle, pct, timeAgo, today, useAuth, useInterval, useToast } from "../lib.jsx";

const FOCUS = 25 * 60, BREAK = 5 * 60;

function Timer() {
  const toast = useToast();
  const [left, setLeft] = useState(FOCUS), [run, setRun] = useState(false), [brk, setBrk] = useState(false);
  useInterval(() => setLeft(l => {
    if (l > 1) return l - 1;
    const nb = !brk; setBrk(nb); toast(nb ? "Focus block done. Take five." : "Break's over. Back to it.", nb ? "☕" : "📚");
    return nb ? BREAK : FOCUS;
  }), run ? 1000 : null);
  const mm = String(Math.floor(left / 60)).padStart(2, "0"), ss = String(left % 60).padStart(2, "0");
  return <span className={"timer" + (run ? (brk ? " brk" : " run") : "")} title={brk ? "Break" : "Focus timer (25 min focus, 5 min break)"}>
    <Icon name={brk ? "Coffee" : "Timer"} size={15} />{mm}:{ss}
    <button className="btn ghost icon" style={{ width: 26, height: 26 }} aria-label={run ? "Pause timer" : "Start timer"} onClick={() => setRun(r => !r)}><Icon name={run ? "Pause" : "Play"} size={14} /></button>
    {(run || left !== FOCUS) && <button className="btn ghost icon" style={{ width: 26, height: 26 }} aria-label="Reset timer" onClick={() => { setRun(false); setBrk(false); setLeft(FOCUS); }}><Icon name="RotateCcw" size={13} /></button>}
  </span>;
}

export default function Reader() {
  const params = useParams();
  const noteId = params["*"].replace(/\/$/, "");
  const nav = useNavigate(), loc = useLocation(), toast = useToast();
  const { theme } = useAuth();
  const [data, setData] = useState(null), [missing, setMissing] = useState(false);
  const [done, setDone] = useState(new Set()), [active, setActive] = useState(null);
  const [panel, setPanel] = useState(false), [tab, setTab] = useState("notes"), [focus, setFocus] = useState(false), [outline, setOutline] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const frame = useRef(null), doneRef = useRef(done), lastAct = useRef(Date.now());
  doneRef.current = done;

  // ---------------------------------------------------------------- load the note
  useEffect(() => {
    let off = false;
    setData(null); setMissing(false); setLoaded(false); setActive(null);
    api(`/api/notes/${noteId}`).then(d => {
      if (off) return;
      setData(d); setDone(new Set(d.me?.done || []));
      api("/api/reading", { method: "POST", body: { noteId } }).catch(() => {});
    }, e => { if (!off && e.status === 404) setMissing(true); });
    return () => { off = true; };
  }, [noteId]);

  // ---------------------------------------------------------------- the bridge the note page talks to
  const saveSection = useRef(null);
  useEffect(() => {
    if (!data) return;
    window.__studyloom = {
      theme: () => document.documentElement.dataset.theme || "light",
      hue: data.subject.hue,
      done: () => [...doneRef.current],
      onDone: ids => {
        const now = new Set(ids), before = doneRef.current;
        const changed = [...new Set([...now, ...before])].filter(id => now.has(id) !== before.has(id));
        setDone(now);
        for (const id of changed) {
          api("/api/progress", { method: "POST", body: { noteId, sectionId: id, done: now.has(id), today: today() } }).catch(() => toast("Couldn't save that. Check your connection.", "⚠️"));
          if (now.has(id)) { const s = data.sections.find(x => x.id === id); toast(`Understood: ${s?.title || "section"}`, "✅"); }
        }
        if (now.size === data.sections.length && changed.some(id => now.has(id))) setTimeout(() => toast("Note finished. Nicely done.", "🎉"), 700);
      },
      onSection: id => {
        setActive(id); lastAct.current = Date.now();
        clearTimeout(saveSection.current);
        saveSection.current = setTimeout(() => api("/api/reading", { method: "POST", body: { noteId, section: id } }).catch(() => {}), 1500);
      },
      onNavigate: (pathname, hash) => {
        const m = pathname.match(/^\/content\/(.+?)\.html$/);
        if (!m) return false;
        const target = m[1];
        const subj = target.match(/^([^/]+)\/index$/);
        nav(subj ? `/app/s/${subj[1]}` : `/app/learn/${target}${hash ? "#" + hash : ""}`);
        return true;
      },
    };
    return () => { delete window.__studyloom; };
  }, [data, noteId, nav, toast]);

  // theme changes reach the page without a reload
  useEffect(() => { try { frame.current?.contentWindow?.__slTheme?.(theme); } catch {} }, [theme]);

  const goto = useCallback(id => { try { frame.current?.contentWindow?.__slGoto?.(id); } catch {} setOutline(false); }, []);
  const onLoad = () => {
    setLoaded(true);
    const w = frame.current?.contentWindow;
    if (w) for (const ev of ["scroll", "pointerdown", "keydown", "mousemove"]) w.addEventListener(ev, () => { lastAct.current = Date.now(); }, { passive: true });
    const want = loc.hash.slice(1) || data?.me?.state?.last_section;
    if (want) setTimeout(() => {
      goto(want);
      if (!loc.hash && data?.me?.state?.last_section) { const s = data.sections.find(x => x.id === want); if (s) toast(`Picked up where you left off: ${s.title}`, "📍"); }
    }, 350);
  };
  useEffect(() => { if (loaded && loc.hash) goto(loc.hash.slice(1)); }, [loc.hash]);   // eslint-disable-line

  // ---------------------------------------------------------------- study time: count only active, visible time
  useInterval(() => {
    if (document.visibilityState === "visible" && Date.now() - lastAct.current < 90_000)
      api("/api/heartbeat", { method: "POST", body: { seconds: 30, today: today() } }).catch(() => {});
  }, data ? 30_000 : null);
  useEffect(() => { const k = () => { lastAct.current = Date.now(); }; addEventListener("keydown", k); addEventListener("pointermove", k); return () => { removeEventListener("keydown", k); removeEventListener("pointermove", k); }; }, []);

  // keyboard: [ previous note, ] next note, F focus mode, N notes panel
  useEffect(() => {
    const k = e => {
      if (e.target.closest?.("input, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "[" && data?.prev) nav(`/app/learn/${data.prev.id}`);
      if (e.key === "]" && data?.next) nav(`/app/learn/${data.next.id}`);
      if (e.key.toLowerCase() === "f") setFocus(f => !f);
      if (e.key.toLowerCase() === "n") setPanel(p => !p);
    };
    addEventListener("keydown", k); return () => removeEventListener("keydown", k);
  }, [data, nav]);

  // ---------------------------------------------------------------- bookmarks and my notes
  const [marks, setMarks] = useState([]), [mine, setMine] = useState([]);
  useEffect(() => { setMarks(data?.me?.bookmarks || []); setMine(data?.me?.notes || []); }, [data]);
  const curSection = active || data?.sections[0]?.id || "";
  const marked = marks.includes(curSection);
  const toggleMark = async () => {
    const on = !marked;
    setMarks(m => on ? [...m, curSection] : m.filter(x => x !== curSection));
    await api("/api/bookmarks", { method: "POST", body: { noteId, sectionId: curSection, on } }).catch(() => {});
    toast(on ? "Bookmarked this section" : "Bookmark removed", on ? "🔖" : "");
  };
  const [draft, setDraft] = useState(""), [editing, setEditing] = useState(null);
  const addNote = async e => {
    e.preventDefault(); if (!draft.trim()) return;
    try { const r = await api("/api/my-notes", { method: "POST", body: { noteId, sectionId: curSection, body: draft } }); setMine(m => [...m, r.note]); setDraft(""); }
    catch (x) { toast(x.message, "⚠️"); }
  };
  const saveEdit = async (id, body) => { const r = await api(`/api/my-notes/${id}`, { method: "PATCH", body: { body } }); setMine(m => m.map(n => n.id === id ? r.note : n)); setEditing(null); };
  const delNote = async id => { await api(`/api/my-notes/${id}`, { method: "DELETE" }); setMine(m => m.filter(n => n.id !== id)); };

  const secTitle = useMemo(() => Object.fromEntries((data?.sections || []).map(s => [s.id, s.title])), [data]);
  if (missing) return <div className="page"><Empty icon="FileQuestion" title="This page isn't in the study path" action={<div className="row"><Link className="btn" to="/app/library">Library</Link><a className="btn" href={`/content/${noteId}.html`} target="_blank" rel="noreferrer">Open it on its own <Icon name="ExternalLink" size={15} /></a></div>}>It exists in the notes, but no subject lists it.</Empty></div>;
  if (!data) return <div className="reader"><div className="rd-outline"><div className="skeleton" style={{ height: 300 }} /></div><div className="rd-center"><div className="rd-bar" /><div className="skeleton" style={{ margin: 30, height: 500 }} /></div><div /></div>;

  const { note, subject, sections } = data;
  const doneCount = sections.filter(s => done.has(s.id)).length;
  const sectionNotes = mine.filter(n => n.section_id === curSection), otherNotes = mine.filter(n => n.section_id !== curSection);

  return <div className={"reader" + (panel ? " panel" : "") + (focus ? " focus" : "") + (outline ? " outline-open" : "")} style={hueStyle(subject.hue)}>
    <aside className="rd-outline" aria-label="Sections of this note">
      <Link className="rd-back" to={`/app/s/${subject.id}`}><Icon name="ArrowLeft" size={15} />{subject.title}</Link>
      <div className="rd-subject"><span className="eyebrow">{note.eyebrow || subject.title}</span><h2>{note.title}</h2></div>
      <div className="rd-progress"><small><span>{doneCount} of {sections.length} understood</span><span>{pct(doneCount, sections.length)}%</span></small><div className="bar"><i style={{ width: pct(doneCount, sections.length) + "%" }} /></div></div>
      <div className="rd-secs">{sections.map((s, i) => <div key={s.id} className="rd-sec-wrap">
        <button className={"rd-sec" + (s.id === active ? " on" : "") + (done.has(s.id) ? " done" : "")} onClick={() => goto(s.id)}>
          <span className="mk">{done.has(s.id) ? <Icon name="Check" size={13} strokeWidth={3} /> : i + 1}</span><span className="grow">{s.title}</span>
          {!!s.questions?.length && <span className="rd-qcount" title={`${s.questions.length} ${s.questions.length === 1 ? "question" : "questions"} in this section`}>{s.questions.length}Q</span>}
          {marks.includes(s.id) && <Icon name="Bookmark" size={13} style={{ color: "var(--accent)" }} />}
        </button>
        {!!s.questions?.length && <div className="rd-qs">{s.questions.map(q => <button key={q.n} className="rd-q" onClick={() => goto("pq-" + q.n)} title={q.text}>
          <span className="qb">Q{q.n}</span><span className="qt">{q.text}</span>
        </button>)}</div>}
      </div>)}</div>
      <div className="rd-others">
        {data.prev && <><h4>Previous</h4><Link to={`/app/learn/${data.prev.id}`}>← {data.prev.title}</Link></>}
        {data.next && <><h4>Next</h4><Link to={`/app/learn/${data.next.id}`}>{data.next.title} →</Link></>}
      </div>
    </aside>

    <div className="rd-center">
      <div className="rd-bar">
        <button className="btn ghost icon" aria-label="Menu" onClick={() => document.querySelector(".side")?.classList.toggle("open")}><Icon name="Menu" size={18} /></button>
        <button className="btn ghost icon" aria-label="Sections" title="Sections" onClick={() => { if (innerWidth <= 1100) setOutline(o => !o); else setFocus(f => !f); }}><Icon name={focus ? "PanelLeftOpen" : "PanelLeftClose"} size={18} /></button>
        <span className="crumb grow">{subject.title} / <b>{note.title}</b>{active && secTitle[active] ? <> / {secTitle[active]}</> : null}</span>
        <Timer />
        <button className="btn ghost icon" aria-pressed={marked} aria-label={marked ? "Remove bookmark" : "Bookmark this section"} title="Bookmark this section" onClick={toggleMark}><Icon name={marked ? "BookmarkCheck" : "Bookmark"} size={18} style={marked ? { color: "var(--accent)" } : null} /></button>
        <button className="btn ghost icon" aria-label="Search" title="Search (Ctrl K)" onClick={() => window.__slOpenSearch?.()}><Icon name="Search" size={18} /></button>
        <button className={"btn sm" + (panel ? " ink" : "")} onClick={() => setPanel(p => !p)}><Icon name="NotebookPen" size={16} />Notes{mine.length ? ` · ${mine.length}` : ""}</button>
        <div className="row" style={{ gap: 2 }}>
          <button className="btn ghost icon" disabled={!data.prev} aria-label="Previous note" title="Previous note ([)" onClick={() => nav(`/app/learn/${data.prev.id}`)}><Icon name="ChevronLeft" size={18} /></button>
          <button className="btn ghost icon" disabled={!data.next} aria-label="Next note" title="Next note (])" onClick={() => nav(`/app/learn/${data.next.id}`)}><Icon name="ChevronRight" size={18} /></button>
        </div>
      </div>
      <div className="rd-frame">
        <iframe key={note.id} ref={frame} src={`/content/${note.path}`} title={note.title} onLoad={onLoad} allow="clipboard-write" />
        <div className={"rd-loading" + (loaded ? " gone" : "")}><div className="row muted"><Icon name="LoaderCircle" className="spin" /> Opening {note.title}…</div></div>
      </div>
    </div>

    <aside className="rd-panel" aria-label="Your notes">
      <div className="rd-panel-in">
        <div className="row"><b className="grow display" style={{ fontSize: 19 }}>Notebook</b><button className="btn ghost icon" aria-label="Close" onClick={() => setPanel(false)}><Icon name="X" size={18} /></button></div>
        <div className="tabs"><button className={tab === "notes" ? "on" : ""} onClick={() => setTab("notes")}>My notes</button><button className={tab === "marks" ? "on" : ""} onClick={() => setTab("marks")}>Bookmarks</button></div>
        {tab === "notes" ? <>
          <form onSubmit={addNote} className="stack" style={{ "--gap": "8px" }}>
            <label className="muted" style={{ fontSize: 13 }} htmlFor="draft">A note on <b style={{ color: "var(--ink)" }}>{secTitle[curSection] || "this note"}</b></label>
            <textarea id="draft" className="input" value={draft} onChange={e => setDraft(e.target.value)} placeholder="What clicked? What to remember? What's still unclear?" maxLength={20000}
              onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) addNote(e); }} />
            <button className="btn primary sm" disabled={!draft.trim()}>Save note <kbd style={{ background: "transparent", color: "inherit", borderColor: "currentColor", opacity: .7 }}>Ctrl ↵</kbd></button>
          </form>
          {[["On this section", sectionNotes], ["Elsewhere in this note", otherNotes]].map(([h, list]) => !!list.length && <div key={h} className="stack" style={{ "--gap": "8px" }}>
            <span className="eyebrow">{h}</span>
            {list.map(n => <div key={n.id} className="mynote">
              {editing === n.id ? <EditNote note={n} onSave={b => saveEdit(n.id, b)} onCancel={() => setEditing(null)} /> : n.body}
              <div className="meta"><button className="btn ghost sm" style={{ height: 24, padding: "0 6px" }} onClick={() => goto(n.section_id)}>{secTitle[n.section_id] || "note"}</button>
                <span>{timeAgo(n.updated_at)}</span>
                <span className="row" style={{ gap: 0 }}><button className="btn ghost icon" style={{ width: 26, height: 26 }} aria-label="Edit" onClick={() => setEditing(n.id)}><Icon name="Pencil" size={13} /></button>
                <button className="btn ghost icon" style={{ width: 26, height: 26 }} aria-label="Delete" onClick={() => delNote(n.id)}><Icon name="Trash2" size={13} /></button></span></div>
            </div>)}
          </div>)}
          {!mine.length && <p className="muted" style={{ fontSize: 14 }}>Notes you write here are saved to your account and shown next to the section they're about.</p>}
        </> : <div className="stack" style={{ "--gap": "6px" }}>
          {marks.length ? marks.map(id => <button key={id} className="rd-sec" onClick={() => goto(id)}><Icon name="Bookmark" size={15} style={{ color: "var(--accent)" }} /><span className="grow">{secTitle[id] || "Whole note"}</span></button>)
            : <p className="muted" style={{ fontSize: 14 }}>Use the bookmark button in the top bar to keep a section close at hand.</p>}
        </div>}
        <p className="faint" style={{ fontSize: 12.5 }}>Shortcuts: <kbd>[</kbd> <kbd>]</kbd> previous / next note · <kbd>F</kbd> focus · <kbd>N</kbd> notebook</p>
      </div>
    </aside>
  </div>;
}

function EditNote({ note, onSave, onCancel }) {
  const [v, setV] = useState(note.body);
  return <div className="stack" style={{ "--gap": "8px" }}><textarea className="input" value={v} onChange={e => setV(e.target.value)} autoFocus />
    <div className="row"><button className="btn primary sm" onClick={() => onSave(v)} disabled={!v.trim()}>Save</button><button className="btn sm" onClick={onCancel}>Cancel</button></div></div>;
}
