// The exam hall: the list of papers, a paper's start screen, the exam room, and the result review.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Empty, Icon, Ring } from "../components.jsx";
import { api, hueStyle, MathHtml, pct, plural, timeAgo, useFetch, useInterval, useToast } from "../lib.jsx";
import "../styles/exams.css";

const TYPE = { MCQ: "Single correct", MSQ: "Multiple correct", NUM: "Numerical answer", TEXT: "Short answer", COMP: "Comprehension" };
const LETTER = i => String.fromCharCode(65 + i);
const answered = r => r != null && (Array.isArray(r) ? r.length > 0 : String(r).trim() !== "");
const fmtClock = s => { s = Math.max(0, Math.floor(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60; return (h ? h + ":" + String(m).padStart(2, "0") : String(m)) + ":" + String(x).padStart(2, "0"); };
const fmtSecs = s => (s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60 ? (s % 60) + " s" : ""}`.trim());
const noteLink = ref => "/app/learn/" + ref;
const marksLabel = m => `${m} ${m === 1 ? "mark" : "marks"}`;

/* ================================================================ the list */
export function ExamHall() {
  const { data } = useFetch("/api/papers");
  const groups = useMemo(() => {
    const p = data?.papers || [];
    return [["Past papers", "Real exam questions, every answer checked by running the code.", p.filter(x => x.source === "pyq")],
            ["Practice papers from the notes", "Every question in a subject's notes, collected into one paper.", p.filter(x => x.source === "notes")]];
  }, [data]);
  return <div className="page">
    <div className="page-head"><div><span className="eyebrow">Exam hall</span><h1 className="display" style={{ marginTop: 8 }}>Sit a paper. Then learn from it.</h1>
      <p>Take a paper against the clock, or practise it one question at a time. Every answer comes with an explanation and a link to the section of the notes that teaches it.</p></div></div>
    {!data ? <div className="skeleton" style={{ height: 260 }} /> : groups.map(([title, sub, list]) => !!list.length && <section key={title}>
      <div className="section-title"><h2>{title}</h2><span className="muted" style={{ fontSize: 14 }}>{sub}</span></div>
      <div className="paper-grid">{list.map(p => <PaperCard key={p.id} p={p} />)}</div>
    </section>)}
    {data && !data.papers.length && <div className="card"><Empty icon="ClipboardCheck" title="No papers yet">Papers appear here once they are added.</Empty></div>}
  </div>;
}

function PaperCard({ p }) {
  const best = p.me?.best != null ? pct(p.me.best, p.max_marks) : null;
  return <Link to={`/app/exams/${p.id}`} className="paper-card rise" style={hueStyle(p.hue)}>
    <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
      <span className="chip tint">{p.course}</span>{p.exam && <span className="chip">{p.exam}</span>}
      {p.me?.open_id && <span className="chip warn"><Icon name="Hourglass" size={13} />In progress</span>}
    </div>
    <h3>{p.title}</h3>
    <p className="pc-desc">{p.description}</p>
    <div className="pc-stats"><span><Icon name="ListChecks" size={15} />{plural(p.questions, "question")}</span><span><Icon name="Trophy" size={15} />{marksLabel(p.max_marks)}</span><span><Icon name="Clock" size={15} />{p.duration} min</span></div>
    {best != null ? <div className="pc-best"><div className="bar"><i style={{ width: best + "%" }} /></div><span>Best {best}%</span></div>
      : <span className="pc-go">Start <Icon name="ArrowRight" size={15} /></span>}
  </Link>;
}

/* ================================================================ the start screen */
export function PaperIntro() {
  const { id } = useParams(), nav = useNavigate(), toast = useToast();
  const list = useFetch("/api/papers"), hist = useFetch(`/api/papers/${id}/attempts`);
  const [mode, setMode] = useState("exam"), [busy, setBusy] = useState(false);
  const p = list.data?.papers.find(x => x.id === id);
  const open = hist.data?.attempts.find(a => !a.submitted_at && a.mode === mode);
  const done = (hist.data?.attempts || []).filter(a => a.submitted_at);
  const start = async fresh => {
    setBusy(true);
    try { const r = await api(`/api/papers/${id}/start`, { method: "POST", body: { mode, fresh } }); nav(`/app/attempts/${r.attempt.id}`); }
    catch (x) { toast(x.message, "⚠️"); setBusy(false); }
  };
  if (list.data && !p) return <div className="page"><div className="card"><Empty icon="SearchX" title="Paper not found" action={<Link className="btn primary" to="/app/exams">Back to the exam hall</Link>}>It may have been renamed.</Empty></div></div>;
  if (!p) return <div className="page"><div className="skeleton" style={{ height: 320 }} /></div>;
  return <div className="page" style={{ maxWidth: 980, ...hueStyle(p.hue) }}>
    <Link to="/app/exams" className="btn ghost sm" style={{ marginBottom: 14 }}><Icon name="ArrowLeft" size={16} />Exam hall</Link>
    <div className="intro-hero card">
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}><span className="chip tint">{p.course}</span>{p.exam && <span className="chip">{p.exam}</span>}</div>
      <h1 className="display">{p.title}</h1>
      <p className="muted">{p.description}</p>
      <div className="intro-stats">
        <div><b>{p.questions}</b><span>questions</span></div><div><b>{p.max_marks}</b><span>marks</span></div><div><b>{p.duration}</b><span>minutes</span></div>
        <div><b>{done.length}</b><span>{done.length === 1 ? "attempt" : "attempts"}</span></div>
      </div>
    </div>

    <h2 className="intro-h">How do you want to take it?</h2>
    <div className="mode-pick" role="radiogroup" aria-label="Mode">
      {[["exam", "Timer", "Exam", `A ${p.duration}-minute countdown. Answers and explanations appear only after you submit, and the paper submits itself when time runs out.`],
        ["practice", "Sparkles", "Practice", "No clock. Check each answer as you go and read the explanation straight away. Your score still counts at the end."]].map(([k, ic, t, d]) =>
        <button key={k} role="radio" aria-checked={mode === k} className={"mode-card" + (mode === k ? " on" : "")} onClick={() => setMode(k)}>
          <span className="mc-ic"><Icon name={ic} size={22} /></span><b>{t}</b><span>{d}</span>
        </button>)}
    </div>

    <div className="intro-start">
      {open ? <>
        <button className="btn primary lg" disabled={busy} onClick={() => start(false)}><Icon name="Play" size={18} />Resume {mode === "exam" ? "exam" : "practice"}</button>
        <button className="btn lg" disabled={busy} onClick={() => start(true)}><Icon name="RotateCcw" size={17} />Start over</button>
        <span className="muted" style={{ fontSize: 14 }}>You started this {timeAgo(open.started_at)}. Your answers were saved.</span>
      </> : <button className="btn primary lg" disabled={busy} onClick={() => start(false)}><Icon name="Play" size={18} />Start {mode === "exam" ? "the exam" : "practising"}</button>}
    </div>

    <div className="intro-cols">
      <section className="card band">
        <h3 className="display" style={{ fontSize: 19, marginBottom: 12 }}>In the room</h3>
        <ul className="intro-list">
          <li><b>Question menu.</b> Jump to any question from the grid. Its colour shows where you are with it.</li>
          <li><b>Mark for review.</b> Flag a question to come back to. A flagged answer is still counted.</li>
          <li><b>Clear selection.</b> Takes an answer back, so the question counts as unanswered.</li>
          <li><b>Reading view.</b> Shows every question on one long page.</li>
          <li><b>Autosave.</b> Every choice saves as you make it. Leave, and you can resume later.</li>
        </ul>
        <div className="legend" style={{ marginTop: 16 }}><Legend /></div>
      </section>
      <section className="card band">
        <h3 className="display" style={{ fontSize: 19, marginBottom: 12 }}>Keyboard</h3>
        <ul className="keys">
          <li><span className="k"><kbd>←</kbd><kbd>→</kbd></span>Previous and next question</li>
          <li><span className="k"><kbd>A</kbd>–<kbd>D</kbd>or<kbd>1</kbd>–<kbd>4</kbd></span>Choose an option</li>
          <li><span className="k"><kbd>F</kbd></span>Mark or unmark for review</li>
          <li><span className="k"><kbd>C</kbd></span>Clear the selection</li>
          <li><span className="k"><kbd>Enter</kbd></span>Check the answer (practice mode)</li>
        </ul>
      </section>
    </div>

    {!!done.length && <>
      <h2 className="intro-h">Your attempts</h2>
      <div className="card band"><div className="list">{done.map(a => <Link key={a.id} to={`/app/attempts/${a.id}`} className="list-item">
        <Ring value={pct(a.score, a.max_score)} size={46} stroke={5} hue={p.hue} />
        <span className="grow"><span className="li-t">{a.score} / {a.max_score} marks</span><span className="li-s">{a.mode === "exam" ? "Exam" : "Practice"} · {timeAgo(a.submitted_at)}</span></span>
        <span className="btn ghost sm">Review<Icon name="ArrowRight" size={15} /></span>
      </Link>)}</div></div>
    </>}
  </div>;
}

function Legend({ counts }) {
  const items = [["answered", "Answered"], ["not-answered", "Not answered"], ["not-visited", "Not visited"], ["flagged", "Marked for review"], ["answered-flagged", "Answered and marked"]];
  return items.map(([k, l]) => <span key={k} className="lg-item"><i className={"pal-dot " + k} />{l}{counts && <b>{counts[k] || 0}</b>}</span>);
}

/* ================================================================ an attempt: the room, or its result */
export function Attempt() {
  const { aid } = useParams();
  const { data, error, setData } = useFetch(`/api/attempts/${aid}`);
  if (error) return <div className="page"><div className="card"><Empty icon="SearchX" title="Attempt not found" action={<Link className="btn primary" to="/app/exams">Back to the exam hall</Link>}>{error.message}</Empty></div></div>;
  if (!data) return <div className="page"><div className="skeleton" style={{ height: 420 }} /></div>;
  return data.attempt.submitted_at ? <Result key={"r" + aid} data={data} /> : <Room key={aid} data={data} onSubmitted={setData} />;
}

function statusOf(q, s) {
  if (!s || !s.visited) return "not-visited";
  const a = answered(s.response);
  if (s.flagged) return a ? "answered-flagged" : "flagged";
  return a ? "answered" : "not-answered";
}

function Room({ data, onSubmitted }) {
  const { attempt, paper, questions } = data;
  const nav = useNavigate(), toast = useToast();
  const byId = useMemo(() => Object.fromEntries(questions.map(q => [q.id, q])), [questions]);
  const [qs, setQs] = useState(questions);                // questions, with answers filled in as they are checked
  const [st, setSt] = useState(() => Object.fromEntries(data.answers.map(a => [a.question_id, a])));
  // reopen where they were on this device; otherwise at the first question still unanswered
  const posKey = "sl-attempt-pos-" + attempt.id;
  const [cur, setCur] = useState(() => {
    const saved = Number(localStorage.getItem(posKey));
    if (Number.isInteger(saved) && saved >= 0 && saved < questions.length) return saved;
    return Math.max(0, questions.findIndex(q => q.type !== "COMP" && !answered(st[q.id]?.response)));
  });
  useEffect(() => { try { localStorage.setItem(posKey, String(cur)); } catch { /* private mode */ } }, [cur, posKey]);
  const [reading, setReading] = useState(false), [summary, setSummary] = useState(false), [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(0), [menu, setMenu] = useState(false);
  const chain = useRef(Promise.resolve()), entered = useRef(Date.now());
  const offset = useMemo(() => Date.parse(attempt.now) - Date.now(), [attempt.now]);
  const [now, setNow] = useState(Date.now());
  useInterval(() => setNow(Date.now()), 1000);
  const left = attempt.deadline ? (Date.parse(attempt.deadline) - (now + offset)) / 1000 : null;
  const elapsed = (now + offset - Date.parse(attempt.started_at.replace(" ", "T") + "Z")) / 1000;

  const save = useCallback((qid, body) => {
    if (byId[qid]?.type === "COMP") return;
    setSaving(n => n + 1);
    chain.current = chain.current.then(() => api(`/api/attempts/${attempt.id}/answers/${qid}`, { method: "PUT", body }))
      .catch(x => { toast(x.message, "⚠️"); if (x.status === 409 && /submitted|Time/.test(x.message)) api(`/api/attempts/${attempt.id}`).then(onSubmitted); })
      .finally(() => setSaving(n => n - 1));
  }, [attempt.id, byId]);  // eslint-disable-line

  // visiting a question marks it visited; leaving it adds the time spent
  const q = qs[cur];
  useEffect(() => {
    entered.current = Date.now();
    if (q.type !== "COMP" && !st[q.id]?.visited) { setSt(s => ({ ...s, [q.id]: { ...s[q.id], visited: 1 } })); save(q.id, {}); }
    return () => { const sec = Math.min(3600, Math.round((Date.now() - entered.current) / 1000)); if (sec && q.type !== "COMP") save(q.id, { seconds: sec }); };
  }, [cur]);  // eslint-disable-line

  const setResp = (qid, response) => { setSt(s => ({ ...s, [qid]: { ...s[qid], visited: 1, response } })); save(qid, { response }); };
  const toggleFlag = qid => { const f = !st[qid]?.flagged; setSt(s => ({ ...s, [qid]: { ...s[qid], visited: 1, flagged: f ? 1 : 0 } })); save(qid, { flagged: f }); };
  const choose = (qq, i) => {
    const s = st[qq.id]; if (s?.checked) return;
    if (qq.type === "MCQ") setResp(qq.id, [i]);
    else { const r = new Set(s?.response || []); r.has(i) ? r.delete(i) : r.add(i); setResp(qq.id, [...r].sort((a, b) => a - b)); }
  };
  const check = async qq => {
    try {
      await chain.current;
      const r = await api(`/api/attempts/${attempt.id}/check/${qq.id}`, { method: "POST" });
      setSt(s => ({ ...s, [qq.id]: { ...s[qq.id], checked: 1, correct: r.correct, marks: r.marks } }));
      setQs(list => list.map(x => (x.id === qq.id ? { ...x, ...r.question } : x)));
    } catch (x) { toast(x.message, "⚠️"); }
  };
  const go = i => { setCur(Math.max(0, Math.min(qs.length - 1, i))); setMenu(false); if (reading) document.getElementById("rq-" + qs[i]?.id)?.scrollIntoView({ behavior: "smooth", block: "start" }); else window.scrollTo({ top: 0 }); };
  const submit = async () => {
    setBusy(true);
    try { const sec = Math.round((Date.now() - entered.current) / 1000); if (sec && q.type !== "COMP") save(q.id, { seconds: sec }); entered.current = Date.now(); await chain.current;
          onSubmitted(await api(`/api/attempts/${attempt.id}/submit`, { method: "POST" })); window.scrollTo({ top: 0 }); }
    catch (x) { toast(x.message, "⚠️"); setBusy(false); }
  };

  // time up: submit for them
  const fired = useRef(false);
  useEffect(() => { if (left != null && left <= 0 && !fired.current) { fired.current = true; toast("Time is up. Your paper was submitted.", "⏰"); submit(); } }, [left]);  // eslint-disable-line

  // keyboard
  useEffect(() => {
    const k = e => {
      if (summary || e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) return;
      const key = e.key.toLowerCase();
      if (e.key === "ArrowRight") { e.preventDefault(); go(cur + 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); go(cur - 1); }
      else if (reading) return;
      else if (key === "f" && q.type !== "COMP") toggleFlag(q.id);
      else if (key === "c" && q.type !== "COMP" && !st[q.id]?.checked) setResp(q.id, q.type === "NUM" || q.type === "TEXT" ? null : []);
      else if (e.key === "Enter" && attempt.mode === "practice" && q.type !== "COMP" && !st[q.id]?.checked && answered(st[q.id]?.response)) check(q);
      else if (q.options.length && (/^[a-h]$/.test(key) || /^[1-8]$/.test(key))) { const i = /\d/.test(key) ? Number(key) - 1 : key.charCodeAt(0) - 97; if (i < q.options.length) choose(q, i); }
    };
    addEventListener("keydown", k); return () => removeEventListener("keydown", k);
  });

  const counts = useMemo(() => { const c = {}; qs.forEach(x => { if (x.type !== "COMP") { const s = statusOf(x, st[x.id]); c[s] = (c[s] || 0) + 1; } }); return c; }, [qs, st]);
  const scorable = qs.filter(x => x.type !== "COMP");
  const nAnswered = scorable.filter(x => answered(st[x.id]?.response)).length;
  const warn = left != null && left <= 300, danger = left != null && left <= 60;
  const parent = q.parent_id ? byId[q.parent_id] : null;
  const children = q.type === "COMP" ? qs.filter(x => x.parent_id === q.id) : [];

  return <div className={"room" + (menu ? " menu-open" : "")}>
    <header className="room-top">
      <Link to={`/app/exams/${paper.id}`} className="btn ghost icon" title="Save and leave" aria-label="Save and leave"><Icon name="ArrowLeft" size={18} /></Link>
      <div className="grow rt-title"><b>{paper.title}</b><span>{attempt.mode === "exam" ? "Exam" : "Practice"} · {nAnswered} of {scorable.length} answered · <span className={"save-dot" + (saving ? " on" : "")}>{saving ? "Saving…" : "Saved"}</span></span></div>
      {left != null ? <div className={"timer" + (warn ? " warn" : "") + (danger ? " danger" : "")} role="timer" aria-label="Time left"><Icon name="Timer" size={17} />{fmtClock(left)}</div>
        : <div className="timer calm" title="Time spent"><Icon name="Clock" size={17} />{fmtClock(elapsed)}</div>}
      <button className={"btn sm" + (reading ? " ink" : "")} onClick={() => setReading(r => !r)}><Icon name={reading ? "Eye" : "Rows3"} size={16} /><span className="hide-s">{reading ? "Exit reading view" : "Reading view"}</span></button>
      <button className="btn ghost icon menu-btn" aria-label="Question menu" onClick={() => setMenu(m => !m)}><Icon name="ListChecks" size={19} /></button>
      <button className="btn primary sm" onClick={() => setSummary(true)}><Icon name="Send" size={15} />Submit</button>
    </header>
    <div className="progress-line"><i style={{ width: pct(nAnswered, scorable.length) + "%" }} /></div>

    <div className="room-body">
      <main className="room-main">
        {reading ? qs.map((x, i) => <div key={x.id} id={"rq-" + x.id} className="card q-card reading-q" onClick={() => setCur(i)}>
          <QView q={x} s={st[x.id]} mode={attempt.mode} reveal={!!st[x.id]?.checked} onChoose={i2 => choose(x, i2)} onNum={v => setResp(x.id, v)}
            onFlag={() => toggleFlag(x.id)} onCheck={() => check(x)} parentNote={x.parent_id ? `Part of the comprehension in question ${byId[x.parent_id].number}` : null} />
        </div>) : <div className="card q-card" key={q.id}>
          {parent && <details className="passage" open><summary><Icon name="BookOpen" size={16} />Comprehension (question {parent.number})</summary><MathHtml className="q-html" html={parent.prompt} /></details>}
          <QView q={q} s={st[q.id]} mode={attempt.mode} reveal={!!st[q.id]?.checked} onChoose={i => choose(q, i)} onNum={v => setResp(q.id, v)} onFlag={() => toggleFlag(q.id)} onCheck={() => check(q)} />
          {q.type === "COMP" && <div className="comp-next"><span className="muted">This passage goes with {children.length === 1 ? "question" : "questions"} {children.map(c => c.number).join(" and ")}. It carries no marks of its own.</span>
            <button className="btn primary sm" onClick={() => go(cur + 1)}>Go to question {children[0]?.number}<Icon name="ArrowRight" size={15} /></button></div>}
          <div className="q-controls">
            <button className="btn sm" disabled={cur === 0} onClick={() => go(cur - 1)}><Icon name="ChevronLeft" size={16} />Previous</button>
            {q.type !== "COMP" && <button className="btn ghost sm" disabled={!answered(st[q.id]?.response) || !!st[q.id]?.checked} onClick={() => setResp(q.id, q.type === "NUM" || q.type === "TEXT" ? null : [])}><Icon name="Eraser" size={16} />Clear selection</button>}
            <span className="grow" />
            {q.type !== "COMP" && <button className={"btn sm" + (st[q.id]?.flagged ? " flag-on" : "")} onClick={() => { if (!st[q.id]?.flagged) toggleFlag(q.id); go(cur + 1); }}><Icon name="Flag" size={15} />{st[q.id]?.flagged ? "Next (marked)" : "Mark for review and next"}</button>}
            {cur < qs.length - 1 ? <button className="btn ink sm" onClick={() => go(cur + 1)}>{q.type === "COMP" ? "Next" : "Save and next"}<Icon name="ChevronRight" size={16} /></button>
              : <button className="btn primary sm" onClick={() => setSummary(true)}>Finish<Icon name="Send" size={15} /></button>}
          </div>
        </div>}
      </main>

      <aside className="palette-side" aria-label="Question menu">
        <div className="ps-head"><b>Question menu</b><button className="btn ghost icon menu-close" aria-label="Close" onClick={() => setMenu(false)}><Icon name="X" size={17} /></button></div>
        <div className="pal-grid">
          {qs.map((x, i) => {
            const s = st[x.id], status = x.type === "COMP" ? "comp" : statusOf(x, s);
            const res = s?.checked ? (s.correct ? " right" : " wrong") : "";
            return <button key={x.id} className={"pal " + status + res + (i === cur ? " cur" : "")} onClick={() => go(i)}
              title={`Question ${x.number}: ${x.type === "COMP" ? "comprehension" : status.replace("-", " ")}`} aria-label={`Question ${x.number}`} aria-current={i === cur ? "true" : undefined}>
              {x.type === "COMP" ? <Icon name="BookOpen" size={14} /> : x.number}
              {s?.checked ? <em>{s.correct ? "✓" : "✗"}</em> : null}
            </button>;
          })}
        </div>
        <div className="legend"><Legend counts={counts} /></div>
        <button className="btn primary" style={{ width: "100%" }} onClick={() => setSummary(true)}><Icon name="Send" size={16} />Submit paper</button>
      </aside>
    </div>
    {menu && <div className="scrim" style={{ display: "block", zIndex: 44 }} onClick={() => setMenu(false)} />}

    {summary && <div className="dlg-bg" onMouseDown={() => !busy && setSummary(false)}>
      <div className="dlg card" role="dialog" aria-modal="true" aria-labelledby="sum-t" onMouseDown={e => e.stopPropagation()}>
        <h2 id="sum-t" className="display">Submit the paper?</h2>
        <p className="muted">{left != null ? `${fmtClock(left)} left on the clock. ` : ""}Once you submit, your answers are final and you'll see the marks and explanations.</p>
        <table className="sum-table"><tbody>
          <tr><td><i className="pal-dot answered" />Answered</td><td>{(counts.answered || 0) + (counts["answered-flagged"] || 0)}</td></tr>
          <tr><td><i className="pal-dot not-answered" />Not answered</td><td>{(counts["not-answered"] || 0) + (counts.flagged || 0)}</td></tr>
          <tr><td><i className="pal-dot not-visited" />Not visited</td><td>{counts["not-visited"] || 0}</td></tr>
          <tr><td><i className="pal-dot flagged" />Marked for review</td><td>{(counts.flagged || 0) + (counts["answered-flagged"] || 0)}</td></tr>
          <tr className="tot"><td>Total questions</td><td>{scorable.length}</td></tr>
        </tbody></table>
        <div className="row" style={{ justifyContent: "flex-end", marginTop: 18 }}>
          <button className="btn" disabled={busy} onClick={() => setSummary(false)}>Keep going</button>
          <button className="btn primary" disabled={busy} onClick={submit}>{busy ? <Icon name="LoaderCircle" className="spin" size={16} /> : <Icon name="Send" size={16} />}Submit</button>
        </div>
      </div>
    </div>}
  </div>;
}

/* one question: prompt, options (or a number box), and once revealed, the answer and why */
function QView({ q, s, mode, reveal, onChoose, onNum, onFlag, onCheck, parentNote, result }) {
  const resp = s?.response;
  const chosen = new Set(Array.isArray(resp) ? resp : []);
  const correct = new Set(Array.isArray(q.answer) ? q.answer : []);
  const [num, setNum] = useState(typeof resp === "string" ? resp : "");
  useEffect(() => { setNum(typeof resp === "string" ? resp : ""); }, [q.id, resp == null]);  // eslint-disable-line
  const locked = reveal || result;
  return <>
    <div className="q-head">
      <span className="q-num">{q.type === "COMP" ? "Comprehension" : `Question ${q.number}`}</span>
      <span className="chip">{TYPE[q.type]}</span>
      {q.type !== "COMP" && <span className="chip">{marksLabel(q.marks)}</span>}
      {parentNote && <span className="chip tint">{parentNote}</span>}
      <span className="grow" />
      {result && <ResultChip s={s} q={q} />}
      {!result && q.type !== "COMP" && <button className={"btn ghost sm flag" + (s?.flagged ? " on" : "")} onClick={e => { e.stopPropagation(); onFlag(); }} aria-pressed={!!s?.flagged}><Icon name="Flag" size={15} />{s?.flagged ? "Marked" : "Mark"}</button>}
    </div>
    <MathHtml className="q-html" html={q.prompt} />
    {q.type === "MSQ" && !locked && <p className="q-hint">Choose every option that is correct.</p>}
    {!!q.options.length && <div className="opts" role={q.type === "MCQ" ? "radiogroup" : "group"}>
      {q.options.map((o, i) => {
        const on = chosen.has(i), good = locked && correct.has(i), bad = locked && on && !correct.has(i);
        return <button key={i} className={"opt" + (on ? " on" : "") + (good ? " good" : "") + (bad ? " bad" : "") + (q.type === "MSQ" ? " multi" : "")} disabled={locked}
          role={q.type === "MCQ" ? "radio" : "checkbox"} aria-checked={on} onClick={e => { e.stopPropagation(); onChoose(i); }}>
          <span className="opt-l">{good ? <Icon name="Check" size={15} /> : bad ? <Icon name="X" size={15} /> : LETTER(i)}</span>
          <MathHtml as="span" className="opt-t q-html" html={o} />
          {locked && on && <span className="opt-you">your answer</span>}
        </button>;
      })}
    </div>}
    {(q.type === "NUM" || q.type === "TEXT") && <div className="num-row">
      <input className="input num" inputMode={q.type === "NUM" ? "decimal" : "text"} placeholder="Type your answer" value={num} disabled={locked} aria-label="Your answer"
        onChange={e => setNum(e.target.value.slice(0, 40))} onBlur={() => onNum?.(num.trim() === "" ? null : num.trim())} onKeyDown={e => { if (e.key === "Enter") e.currentTarget.blur(); }} onClick={e => e.stopPropagation()} />
      {locked && q.answer && <span className="chip good">Answer: {q.type === "TEXT" ? q.answer.texts.join(" / ") : <>{q.answer.value}{q.answer.tol ? ` ± ${q.answer.tol}` : ""}</>}</span>}
    </div>}
    {mode === "practice" && !reveal && !result && q.type !== "COMP" && <div className="check-row">
      <button className="btn primary sm" disabled={!answered(resp)} onClick={e => { e.stopPropagation(); onCheck(); }}><Icon name="CircleCheckBig" size={16} />Check answer</button>
      {!answered(resp) && <span className="muted" style={{ fontSize: 13.5 }}>Answer first, then check.</span>}
    </div>}
    {reveal && !result && <div className={"verdict " + (s?.correct ? "good" : "bad")}><Icon name={s?.correct ? "CircleCheckBig" : "CircleX"} size={18} />{s?.correct ? `Correct: +${marksLabel(q.marks)}` : "Not quite. The answer is marked in green."}</div>}
    {(reveal || result) && q.type !== "COMP" && (q.explanation || q.note_ref) && <div className="why">
      <span className="eyebrow">Why</span>
      {q.explanation && <MathHtml className="q-html" html={q.explanation} />}
      {q.note_ref && <Link className="btn sm" to={noteLink(q.note_ref)}><Icon name="BookOpen" size={15} />Read this in the notes</Link>}
    </div>}
  </>;
}

function ResultChip({ s, q }) {
  if (q.type === "COMP") return null;
  if (!answered(s?.response)) return <span className="chip res skip"><Icon name="CircleDashed" size={14} />Not answered · 0</span>;
  return s.correct ? <span className="chip res good"><Icon name="Check" size={14} />+{q.marks}</span> : <span className="chip res bad"><Icon name="X" size={14} />0 / {q.marks}</span>;
}

/* ================================================================ the result */
function Result({ data }) {
  const { attempt, paper, questions, answers } = data;
  const nav = useNavigate(), toast = useToast();
  const st = useMemo(() => Object.fromEntries(answers.map(a => [a.question_id, a])), [answers]);
  const byId = useMemo(() => Object.fromEntries(questions.map(q => [q.id, q])), [questions]);
  const [filter, setFilter] = useState("all");
  const scorable = questions.filter(q => q.type !== "COMP");
  const kind = q => (!answered(st[q.id]?.response) ? "skipped" : st[q.id]?.correct ? "right" : "wrong");
  const n = { right: 0, wrong: 0, skipped: 0 }; scorable.forEach(q => n[kind(q)]++);
  const flagged = scorable.filter(q => st[q.id]?.flagged).length;
  const spent = answers.reduce((t, a) => t + (a.seconds || 0), 0);
  const p = pct(attempt.score, attempt.max_score);
  const shown = questions.filter(q => filter === "all" ? true : q.type !== "COMP" && (filter === "flagged" ? st[q.id]?.flagged : kind(q) === filter));
  const again = async mode => { try { const r = await api(`/api/papers/${paper.id}/start`, { method: "POST", body: { mode, fresh: true } }); nav(`/app/attempts/${r.attempt.id}`); } catch (x) { toast(x.message, "⚠️"); } };
  const verdict = p >= 85 ? "Excellent work." : p >= 60 ? "A solid attempt." : p >= 40 ? "Getting there." : "A good start: now review.";
  return <div className="page result" style={{ maxWidth: 1000 }}>
    <div className="row no-print" style={{ marginBottom: 14 }}>
      <Link to={`/app/exams/${paper.id}`} className="btn ghost sm"><Icon name="ArrowLeft" size={16} />{paper.title}</Link>
    </div>
    <div className="res-hero card">
      <Ring value={p} size={132} stroke={11} label={`${p}% scored`} />
      <div className="grow">
        <span className="eyebrow">{attempt.mode === "exam" ? "Exam" : "Practice"} · submitted {timeAgo(attempt.submitted_at)}</span>
        <h1 className="display">{attempt.score} <span className="muted">/ {attempt.max_score} marks</span></h1>
        <p className="muted">{verdict} {n.wrong + n.skipped ? `Review the ${n.wrong + n.skipped} you missed below: each one links to the section that teaches it.` : "Every question right."}</p>
        <div className="res-stats">
          <span className="good"><Icon name="CircleCheckBig" size={16} />{n.right} correct</span>
          <span className="bad"><Icon name="CircleX" size={16} />{n.wrong} wrong</span>
          <span><Icon name="CircleDashed" size={16} />{n.skipped} not answered</span>
          <span><Icon name="Clock" size={16} />{fmtSecs(spent)}</span>
        </div>
      </div>
    </div>
    <div className="row no-print" style={{ flexWrap: "wrap", margin: "18px 0 26px" }}>
      <button className="btn primary" onClick={() => again("exam")}><Icon name="Timer" size={16} />Retake as an exam</button>
      <button className="btn" onClick={() => again("practice")}><Icon name="Sparkles" size={16} />Practise it</button>
      <button className="btn ghost" onClick={() => print()}><Icon name="Printer" size={16} />Download PDF</button>
    </div>

    <div className="res-filter no-print" role="tablist">
      {[["all", "All", questions.length], ["wrong", "Wrong", n.wrong], ["skipped", "Not answered", n.skipped], ["right", "Correct", n.right], ["flagged", "Marked", flagged]].map(([k, l, c]) =>
        <button key={k} role="tab" aria-selected={filter === k} className={filter === k ? "on" : ""} onClick={() => setFilter(k)}>{l}<b>{c}</b></button>)}
    </div>
    <div className="stack" style={{ "--gap": "14px" }}>
      {shown.map(q => <div key={q.id} className={"card q-card review " + (q.type === "COMP" ? "comp" : kind(q))}>
        <QView q={q} s={st[q.id]} mode={attempt.mode} result parentNote={q.parent_id ? `Part of the comprehension in question ${byId[q.parent_id]?.number}` : null} />
        {q.type !== "COMP" && !!st[q.id]?.seconds && <div className="q-time"><Icon name="Clock" size={13} />{fmtSecs(st[q.id].seconds)} on this question</div>}
      </div>)}
      {!shown.length && <div className="card"><Empty icon="CircleCheckBig" title="Nothing here">No questions match this filter.</Empty></div>}
    </div>
  </div>;
}
