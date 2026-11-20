import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Icon, Logo } from "../components.jsx";
import { api, useAuth } from "../lib.jsx";

export default function Auth({ mode }) {
  const signup = mode === "signup";
  const { setUser } = useAuth();
  const nav = useNavigate(), loc = useLocation();
  const [f, setF] = useState({ name: "", email: "", password: "" });
  const [err, setErr] = useState(null), [busy, setBusy] = useState(false), [show, setShow] = useState(false);
  const set = k => e => setF({ ...f, [k]: e.target.value });
  const submit = async e => {
    e.preventDefault(); setBusy(true); setErr(null);
    try {
      const r = await api(signup ? "/api/auth/signup" : "/api/auth/login", { method: "POST", body: signup ? f : { email: f.email, password: f.password } });
      setUser(r.user);
      nav(loc.state?.from || "/app", { replace: true });
    } catch (x) { setErr(x); } finally { setBusy(false); }
  };
  const fieldErr = k => err?.field === k ? err.message : null;
  return <div className="auth">
    <div className="auth-form">
      <form onSubmit={submit} noValidate>
        <Logo />
        <div>
          <h1 className="display">{signup ? "Make your study desk" : "Welcome back"}</h1>
          <p className="muted" style={{ marginTop: 8 }}>{signup ? "Free, and your progress follows you to any device." : "Pick up exactly where you left off."}</p>
        </div>
        {err && !err.field && <div className="alert" role="alert">{err.message}</div>}
        {signup && <div className="field"><label htmlFor="name">Your name</label>
          <input id="name" className="input" autoComplete="name" value={f.name} onChange={set("name")} aria-invalid={!!fieldErr("name")} required maxLength={80} />
          {fieldErr("name") && <span className="err">{fieldErr("name")}</span>}</div>}
        <div className="field"><label htmlFor="email">Email</label>
          <input id="email" className="input" type="email" autoComplete="email" value={f.email} onChange={set("email")} aria-invalid={!!fieldErr("email")} required />
          {fieldErr("email") && <span className="err">{fieldErr("email")}</span>}</div>
        <div className="field"><label htmlFor="password">Password</label>
          <div style={{ position: "relative" }}>
            <input id="password" className="input" type={show ? "text" : "password"} autoComplete={signup ? "new-password" : "current-password"} value={f.password} onChange={set("password")} aria-invalid={!!fieldErr("password")} required minLength={signup ? 8 : 1} style={{ paddingRight: 48 }} />
            <button type="button" className="btn ghost icon" onClick={() => setShow(s => !s)} aria-label={show ? "Hide password" : "Show password"} style={{ position: "absolute", right: 4, top: 4 }}><Icon name={show ? "EyeOff" : "Eye"} size={18} /></button>
          </div>
          {signup && !fieldErr("password") && <span className="muted" style={{ fontSize: 13 }}>At least 8 characters.</span>}
          {fieldErr("password") && <span className="err">{fieldErr("password")}</span>}</div>
        <button className="btn primary lg" disabled={busy}>{busy ? "One moment…" : signup ? "Create account" : "Sign in"}</button>
        <p className="muted" style={{ textAlign: "center", fontSize: 14.5 }}>
          {signup ? <>Already have an account? <Link to="/login">Sign in</Link></> : <>New here? <Link to="/signup">Create an account</Link></>}
        </p>
      </form>
    </div>
    <div className="auth-art" aria-hidden="true">
      <svg className="weave" viewBox="0 0 600 800" preserveAspectRatio="xMidYMid slice">
        {Array.from({ length: 14 }, (_, i) => <path key={i} d={`M-20 ${60 + i * 52} C 150 ${20 + i * 52}, 300 ${120 + i * 52}, 620 ${50 + i * 52}`} fill="none"
          stroke={["#8f86ff", "#f2a35c", "#5cc581", "#6fd0e0"][i % 4]} strokeOpacity={.18 + (i % 3) * .08} strokeWidth="2" />)}
      </svg>
      <blockquote>“One section, understood, is worth more than a chapter skimmed.”</blockquote>
      <p>Studyloom keeps track of what you've understood, so every session starts in the right place.</p>
    </div>
  </div>;
}
