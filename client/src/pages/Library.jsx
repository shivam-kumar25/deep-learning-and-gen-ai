import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Empty, Icon, Ring, SubjectIcon } from "../components.jsx";
import { hueStyle, pct, plural, useFetch } from "../lib.jsx";

export function Library() {
  const { data } = useFetch("/api/catalogue");
  const [filter, setFilter] = useState("all");
  const rows = useMemo(() => !data ? [] : data.subjects.map(s => {
    const ns = data.notes.filter(n => n.subject_id === s.id);
    const total = ns.reduce((a, n) => a + n.sections, 0), done = ns.reduce((a, n) => a + Math.min(n.done, n.sections), 0);
    return { ...s, notes: ns.length, total, done, minutes: ns.reduce((a, n) => a + n.minutes, 0), questions: ns.reduce((a, n) => a + n.questions, 0) };
  }), [data]);
  const shown = rows.filter(s => filter === "all" || (filter === "started" ? s.done > 0 && s.done < s.total : filter === "done" ? s.total && s.done === s.total : s.done === 0));
  const tracks = TRACKS.map(t => ({ ...t, subjects: shown.filter(s => (s.track || "theory") === t.id) })).filter(t => t.subjects.length);
  const card = s => { const p = pct(s.done, s.total); return <Link key={s.id} to={`/app/s/${s.id}`} className="subject-card" style={hueStyle(s.hue)}>
    <div className="row"><SubjectIcon subject={s} /><span className="chip tint">{s.num}</span><span className="grow" /><Ring value={p} size={46} stroke={5} hue={s.hue} /></div>
    <h3>{s.title}</h3>
    <p className="muted" style={{ fontSize: 14, lineHeight: 1.5 }}>{s.blurb}</p>
    <div className="meta"><span>{plural(s.notes, "note")} · ~{Math.round(s.minutes / 60) || 1} h</span><span>{plural(s.questions, "question")}</span></div>
  </Link>; };
  return <div className="page">
    <div className="page-head"><div><span className="eyebrow">Library</span><h1 className="display" style={{ marginTop: 8 }}>Neurons first, then PyTorch, then architectures and projects.</h1>
      <p>They're ordered so each one builds on the ones before. Jump in anywhere: your progress is kept per section.</p></div>
      <div className="tabs" role="tablist" style={{ minWidth: 340 }}>{[["all", "All"], ["started", "In progress"], ["new", "Not started"], ["done", "Finished"]].map(([k, l]) =>
        <button key={k} role="tab" aria-selected={filter === k} className={filter === k ? "on" : ""} onClick={() => setFilter(k)}>{l}</button>)}</div>
    </div>
    {!data ? <div className="grid-subjects">{Array.from({ length: 6 }, (_, i) => <div key={i} className="skeleton" style={{ height: 210 }} />)}</div> :
      tracks.length ? tracks.map(t => <section className="group" key={t.id}>
        <div className="group-h"><h2>{t.title}</h2><p className="muted" style={{ margin: "4px 0 0" }}>{t.blurb}</p></div>
        <div className="grid-subjects">{t.subjects.map(card)}</div>
      </section>) : <Empty icon="Filter" title="Nothing here yet">No subject matches this filter.</Empty>}
  </div>;
}

const TRACKS = [
  { id: "theory", title: "How neural networks learn", blurb: "From a single neuron to backpropagation: the ideas every model in this course is built on. Start here." },
  { id: "pytorch", title: "PyTorch", blurb: "The framework: tensors, autograd and the training workflow, run live on this computer." },
  { id: "architectures", title: "Architectures", blurb: "Convolutional networks for images, from the convolution operation to ResNet." },
  { id: "projects", title: "Projects", blurb: "End-to-end deep learning on real data: images, tables, transfer learning and segmentation." },
];

export function Subject() {
  const { id } = useParams();
  const { data } = useFetch("/api/catalogue");
  if (!data) return <div className="page"><div className="skeleton" style={{ height: 220 }} /></div>;
  const s = data.subjects.find(x => x.id === id);
  if (!s) return <div className="page"><Empty icon="SearchX" title="No such subject" action={<Link className="btn" to="/app/library">Back to the library</Link>}>It may have been renamed.</Empty></div>;
  const notes = data.notes.filter(n => n.subject_id === id);
  const groups = [];
  for (const n of notes) { const g = groups.at(-1); if (g && g.title === n.group_title) g.notes.push(n); else groups.push({ title: n.group_title, notes: [n] }); }
  const total = notes.reduce((a, n) => a + n.sections, 0), done = notes.reduce((a, n) => a + Math.min(n.done, n.sections), 0);
  const next = notes.find(n => n.done < n.sections) || notes[0];
  return <div className="page" style={hueStyle(s.hue)}>
    <Link className="rd-back" to="/app/library"><Icon name="ArrowLeft" size={16} /> Library</Link>
    <section className="subject-hero rise" style={{ marginTop: 10 }}>
      <div className="row"><SubjectIcon subject={s} /><span className="eyebrow" style={{ color: "var(--tint-ink)" }}>{`Subject ${s.num}`}</span></div>
      <h1 className="display">{s.title}</h1>
      <p>{s.blurb}</p>
      <div className="row" style={{ marginTop: 22, flexWrap: "wrap", gap: 18 }}>
        {next && <Link className="btn primary" to={`/app/learn/${next.id}`} style={{ "--brand": "var(--tint-strong)", "--brand-2": "var(--tint-strong)" }}>{done ? "Continue" : "Start"}: {next.title} <Icon name="ArrowRight" size={17} /></Link>}
        <div style={{ minWidth: 220 }}><div className="bar"><i style={{ width: pct(done, total) + "%" }} /></div><small className="muted">{done} of {total} sections understood</small></div>
      </div>
    </section>
    {groups.map(g => <section className="group" key={g.title}>
      {groups.length > 1 && <div className="group-h"><h2>{g.title}</h2></div>}
      {g.notes.map((n, i) => { const full = n.sections && n.done >= n.sections;
        return <Link key={n.id} to={`/app/learn/${n.id}`} className={"note-row" + (full ? " done" : "")}>
          <span className="n">{full ? <Icon name="Check" size={20} /> : (n.id.match(/(\d\d)[-_][^/]*$/)?.[1] || (n.kind === "lab" ? <Icon name="FlaskConical" size={19} /> : i + 1))}</span>
          <span style={{ minWidth: 0 }}><h3>{n.title}</h3><p>{n.lead || n.eyebrow}</p></span>
          <span className="side-meta">
            {n.kind === "lab" && <span className="chip">lab</span>}
            {n.runnable > 0 && <span className="chip" title="runnable examples"><Icon name="Play" size={12} />{n.runnable}</span>}
            {n.widgets > 0 && <span className="chip" title="interactive demos"><Icon name="MousePointerClick" size={12} />{n.widgets}</span>}
            <span className="chip">{n.minutes} min</span>
            <span className="chip tint">{Math.min(n.done, n.sections)}/{n.sections}</span>
          </span>
        </Link>; })}
    </section>)}
  </div>;
}
