import { useState } from "react";
import { Link } from "react-router-dom";
import { Icon, Logo, SubjectIcon } from "../components.jsx";
import { hueStyle, useAuth, useFetch } from "../lib.jsx";
import "../styles/landing.css";

const FAQ = [
  ["What is Studyloom?", "A place to study deep learning: from the neuron and the perceptron to backpropagation, PyTorch, convolutional networks, transfer learning and U-Net. Every code cell runs on a live Python kernel with PyTorch on this computer, so you can change the code and run it again."],
  ["Do I need to install anything?", "The app runs the code on this computer's Python environment (the project's .venv, with PyTorch and CUDA if you have an NVIDIA GPU). Cells share their variables like a notebook; the chip at the bottom left restarts the kernel. If the kernel is switched off, cells fall back to Python in the browser, without PyTorch."],
  ["What does an account give me?", "Your progress section by section, a daily goal and streak, bookmarks, your own notes on any section, and search across everything. It follows you to any device you sign in on."],
  ["What do you store about me?", "Your name, email, a hashed password, and your study activity. You can download all of it as JSON or delete your account at any time from Settings: deleting removes everything."],
  ["Is it free?", "Yes."],
];

export default function Landing() {
  const { user } = useAuth();
  const stats = useFetch("/api/stats").data;
  const cat = useFetch("/api/catalogue").data;
  const [open, setOpen] = useState(0);
  const counts = {};
  cat?.notes.forEach(n => { counts[n.subject_id] = (counts[n.subject_id] || 0) + 1; });
  const cta = user ? <Link className="btn primary lg" to="/app">Go to your desk <Icon name="ArrowRight" size={18} /></Link>
                   : <Link className="btn primary lg" to="/signup">Start studying, free <Icon name="ArrowRight" size={18} /></Link>;
  return <div className="landing">
    <header className="l-nav">
      <Logo />
      <nav aria-label="Sections"><a href="#curriculum">Curriculum</a><a href="#features">Features</a><a href="#how">How it works</a><a href="#faq">FAQ</a></nav>
      <div className="row">{user ? <Link className="btn primary sm" to="/app">Open Studyloom</Link> : <><Link className="btn ghost sm" to="/login">Sign in</Link><Link className="btn primary sm" to="/signup">Get started</Link></>}</div>
    </header>

    {/* ------------------------------------------------ hero */}
    <section className="l-hero">
      <div className="l-hero-copy rise">
        <span className="pill"><span className="pulse" /> Neurons · backprop · PyTorch · CNNs · projects</span>
        <h1 className="display">Learn deep learning, <em>one runnable cell</em> at a time.</h1>
        <p className="lead">From a single neuron to a ResNet you fine-tune yourself. Read a section, run its code on a live PyTorch kernel, change a number and run it again, mark it understood, and pick up exactly where you left off tomorrow.</p>
        <div className="row l-cta">{cta}<a className="btn lg" href="#curriculum">See the curriculum</a></div>
        <p className="tiny muted"><Icon name="ShieldCheck" size={15} /> Free · no install · your data stays yours</p>
      </div>
      <div className="l-hero-art" aria-hidden="true">
        <div className="mock reader-mock" style={hueStyle(188)}>
          <div className="mock-bar"><i /><i /><i /><span>MLPs · solving XOR</span></div>
          <div className="mock-body">
            <div className="mock-band" />
            <div className="mock-h"><span className="mock-num">03</span><b>A hidden layer solves XOR</b></div>
            <p>Two hidden units draw two lines; the output keeps the band between them, which no single line could do.</p>
            <div className="mock-code">
              <div className="mc-head"><span>python · live kernel</span><span className="mc-run">▶ run</span></div>
              <pre>model = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(),{"\n"}                      nn.Linear(2, 1)).to("cuda"){"\n"}train(model, X, y, epochs=5000)</pre>
              <div className="mc-out">predictions [0, 1, 1, 0] · loss 0.0018</div>
            </div>
            <div className="mock-done"><span>✓ Understood</span></div>
          </div>
        </div>
        <div className="mock float-a"><div className="ring-mock" style={{ "--p": 68 }}><span>68%</span></div><div><b>Neural Network Foundations</b><small>3 of 4 notes</small></div></div>
        <div className="mock float-b"><span className="flame">🔥</span><div><b>12-day streak</b><small>Today: 34 of 30 min</small></div></div>
        <div className="mock float-c"><Icon name="Search" size={15} /><span>“dying ReLU”</span><kbd>Ctrl K</kbd></div>
      </div>
    </section>

    {/* ------------------------------------------------ numbers, live from the database */}
    <section className="l-stats">
      {[["notes", stats?.notes, "notes"], ["sections", stats?.sections, "sections"], ["questions", stats?.questions, "practice questions"]].map(([k, v, l]) =>
        <div key={k}><b className="display">{v != null ? v.toLocaleString() : "—"}</b><span>{l}</span></div>)}
    </section>

    {/* ------------------------------------------------ curriculum */}
    <section className="l-curr" id="curriculum">
      <div className="l-head"><span className="eyebrow">The curriculum</span><h2 className="display">Five subjects, in the order they build on each other.</h2>
        <p>Start with how a network learns, then the PyTorch that builds one, then the convolutional architectures and the projects that put it all to work. Every subject is a path of notes, each split into sections you can finish in a sitting.</p></div>
      <div className="curr-grid">
        {(cat?.subjects || Array.from({ length: 5 }, (_, i) => ({ id: i, num: String(i).padStart(2, "0"), title: "…", hue: 30 * i, icon: "BookOpen", blurb: "" }))).map(s =>
          <Link key={s.id} to={user ? `/app/s/${s.id}` : "/signup"} className="curr" style={hueStyle(s.hue)}>
            <div className="row"><SubjectIcon subject={s} /><span className="curr-num">{s.num}</span></div>
            <h3>{s.title}</h3>
            <p>{s.blurb}</p>
            <span className="curr-foot">{counts[s.id] || 0} notes <Icon name="ArrowUpRight" size={16} /></span>
          </Link>)}
      </div>
    </section>

    {/* ------------------------------------------------ features (dark band) */}
    <section className="l-features" id="features">
      <div className="l-head light"><span className="eyebrow">Not just reading</span><h2 className="display">Every idea comes with code you can run.</h2>
        <p>Code cells run on a real Python kernel with PyTorch on this computer, on the GPU when there is one. Change anything and run it again; add a cell of your own to try an idea.</p></div>
      <div className="feat-grid">
        {[["Play", "A live PyTorch kernel", "Cells share their variables like a notebook and stream their output as they run, so you can watch a training loop's loss fall. Stop a long run at any time."],
          ["ChartColumn", "Figures you can rebuild", "Every diagram, from the perceptron's line to the activation functions, is drawn by code on the page. Change it and see what moves."],
          ["ClipboardCheck", "Checked by running", "Every cell was run while the notes were built, so the outputs you see are the ones the code produces."],
          ["CircleCheckBig", "Progress you can see", "Mark each section understood. Your desk shows what's next, how far each subject is, and your streak."],
          ["NotebookPen", "Your notes, on any section", "Write a note next to the section it belongs to. Find it later from My notes or from the section itself."],
          ["Search", "Search everything", "Ctrl K searches every section of every note at once and takes you to the exact one."]].map(([ic, t, d]) =>
          <div className="feat" key={t}><span className="feat-ic"><Icon name={ic} size={22} /></span><h3>{t}</h3><p>{d}</p></div>)}
      </div>
    </section>

    {/* ------------------------------------------------ how a session works */}
    <section className="l-how" id="how">
      <div className="l-head"><span className="eyebrow">How it works</span><h2 className="display">A study session, start to finish.</h2></div>
      <ol className="steps">
        <li><span className="step-n">1</span><h3>Open your desk</h3><p>Today's goal, your streak, and the note you were in the middle of: one click and you're back on the exact section.</p></li>
        <li><span className="step-n">2</span><h3>Read, try, repeat</h3><p>Each section is short and self-contained. Run its code, change a number, run it again, start the focus timer if you like.</p></li>
        <li><span className="step-n">3</span><h3>Mark it understood</h3><p>Ticking a section saves it to your account. Your rings fill up, your streak grows, and tomorrow starts where today ended.</p></li>
      </ol>
    </section>

    {/* ------------------------------------------------ FAQ */}
    <section className="l-faq" id="faq">
      <div className="l-head"><span className="eyebrow">Questions</span><h2 className="display">Good to know.</h2></div>
      <div className="faq">
        {FAQ.map(([q, a], i) => <div key={q} className={"faq-item" + (open === i ? " open" : "")}>
          <button aria-expanded={open === i} onClick={() => setOpen(open === i ? -1 : i)}><span>{q}</span><Icon name="Plus" size={20} /></button>
          <div className="faq-a" hidden={open !== i}><p>{a}</p></div>
        </div>)}
      </div>
    </section>

    <section className="l-final">
      <h2 className="display">Your desk is ready.</h2>
      <p>Pick a subject, read one section, run one cell. That's a good first day.</p>
      {cta}
    </section>

    <footer className="l-foot">
      <Logo />
      <p className="muted">Notes on deep learning: neurons and perceptrons, activations, losses, gradient descent, backpropagation, PyTorch, CNNs, transfer learning and segmentation.</p>
      <nav className="row"><Link to="/login">Sign in</Link><Link to="/signup">Create account</Link><a href="#faq">FAQ</a></nav>
    </footer>
  </div>;
}
