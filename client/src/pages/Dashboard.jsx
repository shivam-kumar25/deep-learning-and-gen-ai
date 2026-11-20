import { Link } from "react-router-dom";
import { Empty, Heatmap, Icon, Ring, SubjectIcon } from "../components.jsx";
import { hueStyle, minutes, pct, plural, timeAgo, today, useAuth, useFetch } from "../lib.jsx";

function greeting() { const h = new Date().getHours(); return h < 5 ? "Burning the midnight oil" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening"; }

export default function Dashboard() {
  const { user } = useAuth();
  const day = today();
  const { data, loading } = useFetch(`/api/dashboard?today=${day}`);
  if (loading && !data) return <div className="page"><div className="skeleton" style={{ height: 260 }} /><div className="grid-subjects" style={{ marginTop: 24 }}>{Array.from({ length: 6 }, (_, i) => <div key={i} className="skeleton" style={{ height: 150 }} />)}</div></div>;
  if (!data) return <div className="page"><Empty icon="CloudOff" title="Couldn't load your desk">Check your connection and reload the page.</Empty></div>;
  const goalPct = pct(data.today.seconds, data.today.goal);
  const cont = data.recent[0];
  const first = !data.recent.length;
  return <div className="page">
    <section className="band today-hero rise">
      <div>
        <span className="eyebrow">{new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" })}</span>
        <h1 className="display" style={{ marginTop: 10 }}>{greeting()}, {user.name.split(" ")[0]}.</h1>
        <p className="lead">{first ? "Your desk is set up. Pick a first note below, or start right at the beginning of the curriculum." :
          goalPct >= 100 ? "Today's goal is done. Anything more is a bonus." : `${minutes(Math.max(0, data.today.goal - data.today.seconds))} to go for today's goal.`}</p>
        {cont ? <Link className="continue" to={`/app/learn/${cont.note_id}${cont.last_section ? "#" + cont.last_section : ""}`} style={hueStyle(cont.hue)}>
          <span className="subject-icon"><Icon name={cont.icon} size={22} /></span>
          <span className="grow"><small>Continue · {cont.subject_title}</small><b>{cont.title}</b><small>{cont.done} of {cont.sections} sections · opened {timeAgo(cont.opened_at)}</small></span>
          <Icon name="ArrowRight" />
        </Link> : data.next && <Link className="continue" to={`/app/learn/${data.next.id}`} style={hueStyle(data.next.hue)}>
          <span className="subject-icon"><Icon name={data.next.icon} size={22} /></span>
          <span className="grow"><small>Start here · {data.next.subject_title}</small><b>{data.next.title}</b><small>about {data.next.minutes} min</small></span>
          <Icon name="ArrowRight" />
        </Link>}
      </div>
      <div className="goal">
        <Ring value={goalPct} size={132} stroke={11} hue={goalPct >= 100 ? 142 : 250} label={`Daily goal ${goalPct}% done`} />
        <span className="muted" style={{ fontSize: 14 }}>{minutes(data.today.seconds)} of {minutes(data.today.goal)} today</span>
        <div className="stats">
          <div className="stat"><b>{data.streak}</b><small>day streak {data.streak > 0 ? "🔥" : ""}</small></div>
          <div className="stat"><b>{data.totals.sections}</b><small>sections done</small></div>
          <div className="stat"><b>{data.totals.notes}</b><small>notes started</small></div>
        </div>
      </div>
    </section>

    {data.next && cont && data.next.id !== cont.note_id && <Link to={`/app/learn/${data.next.id}`} className="card row" style={{ ...hueStyle(data.next.hue), marginTop: 14, padding: "14px 18px", textDecoration: "none", color: "var(--ink)" }}>
      <Icon name="Sparkles" size={18} style={{ color: "var(--tint-strong)" }} /><span className="grow"><span className="muted">Up next in {data.next.subject_title}: </span><b>{data.next.title}</b></span><Icon name="ArrowRight" size={18} />
    </Link>}

    <div className="section-title"><h2>Your subjects</h2><Link to="/app/library">Open the library →</Link></div>
    <div className="grid-subjects">
      {data.subjects.map(s => { const p = pct(s.done, s.total); return <Link key={s.id} to={`/app/s/${s.id}`} className="subject-card" style={hueStyle(s.hue)}>
        <div className="row"><SubjectIcon subject={s} /><span className="grow" /><Ring value={p} size={46} stroke={5} hue={s.hue} /></div>
        <h3>{s.title}</h3>
        <div><div className="bar"><i style={{ width: p + "%" }} /></div><div className="meta" style={{ marginTop: 8 }}><span>{s.done} / {s.total} sections</span><span>{s.num}</span></div></div>
      </Link>; })}
    </div>

    <div className="two" style={{ marginTop: 16 }}>
      <section className="card band">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 16 }}><h2 className="display" style={{ fontSize: 22 }}>Your last six months</h2><span className="muted" style={{ fontSize: 13 }}>darker = more study</span></div>
        <Heatmap days={data.days} today={day} />
      </section>
      <section className="card band">
        <h2 className="display" style={{ fontSize: 22, marginBottom: 6 }}>Recently opened</h2>
        {data.recent.length ? <div className="list">{data.recent.map(r => <Link key={r.note_id} className="list-item" to={`/app/learn/${r.note_id}`} style={hueStyle(r.hue)}>
          <span className="subject-icon sm"><Icon name={r.icon} size={17} /></span>
          <span className="grow"><span className="li-t">{r.title}</span><span className="li-s">{r.subject_title} · {r.done}/{r.sections} · {timeAgo(r.opened_at)}</span></span>
        </Link>)}</div> : <p className="muted" style={{ marginTop: 10 }}>Notes you open will show up here.</p>}
      </section>
    </div>

    {!!data.bookmarks.length && <>
      <div className="section-title"><h2>Bookmarked</h2><Link to="/app/bookmarks">All bookmarks →</Link></div>
      <div className="row" style={{ flexWrap: "wrap" }}>{data.bookmarks.map(b => <Link key={b.note_id + b.section_id} className="chip tint" style={{ ...hueStyle(b.hue), height: 34, textDecoration: "none" }} to={`/app/learn/${b.note_id}${b.section_id ? "#" + b.section_id : ""}`}><Icon name="Bookmark" size={14} />{b.section_title || b.title}</Link>)}</div>
    </>}
    <p className="faint" style={{ marginTop: 40, fontSize: 13 }}>{plural(data.totals.sections, "section")} understood so far. Press <kbd>Ctrl K</kbd> anywhere to search.</p>
  </div>;
}
