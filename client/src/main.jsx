import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces/opsz.css";
import "@fontsource-variable/jetbrains-mono";
import "./styles/app.css";
import "./styles/shell.css";
import { AuthProvider, ToastProvider, useAuth } from "./lib.jsx";
import { AppShell } from "./components.jsx";
import Landing from "./pages/Landing.jsx";
import Auth from "./pages/Auth.jsx";

const Dashboard = lazy(() => import("./pages/Dashboard.jsx"));
const Reader = lazy(() => import("./pages/Reader.jsx"));
const lib = () => import("./pages/Library.jsx"), acc = () => import("./pages/Account.jsx");
const Library = lazy(() => lib().then(m => ({ default: m.Library })));
const Subject = lazy(() => lib().then(m => ({ default: m.Subject })));
const Bookmarks = lazy(() => acc().then(m => ({ default: m.Bookmarks })));
const MyNotes = lazy(() => acc().then(m => ({ default: m.MyNotes })));
const Settings = lazy(() => acc().then(m => ({ default: m.Settings })));
const NotFound = lazy(() => acc().then(m => ({ default: m.NotFound })));
const ex = () => import("./pages/Exams.jsx");
const ExamHall = lazy(() => ex().then(m => ({ default: m.ExamHall })));
const PaperIntro = lazy(() => ex().then(m => ({ default: m.PaperIntro })));
const Attempt = lazy(() => ex().then(m => ({ default: m.Attempt })));

function Gate({ children, guest }) {
  const { user } = useAuth(), loc = useLocation();
  if (user === undefined) return <div className="page"><div className="skeleton" style={{ height: 240 }} /></div>;
  if (guest) return user ? <Navigate to="/app" replace /> : children;
  return user ? children : <Navigate to="/login" replace state={{ from: loc.pathname + loc.hash }} />;
}

createRoot(document.getElementById("root")).render(<StrictMode>
  <BrowserRouter>
    <AuthProvider><ToastProvider>
      <Suspense fallback={<div className="page"><div className="skeleton" style={{ height: 240 }} /></div>}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/signup" element={<Gate guest><Auth mode="signup" /></Gate>} />
          <Route path="/login" element={<Gate guest><Auth mode="login" /></Gate>} />
          <Route path="/app" element={<Gate><AppShell /></Gate>}>
            <Route index element={<Dashboard />} />
            <Route path="library" element={<Library />} />
            <Route path="s/:id" element={<Subject />} />
            <Route path="learn/*" element={<Reader />} />
            <Route path="exams" element={<ExamHall />} />
            <Route path="exams/:id" element={<PaperIntro />} />
            <Route path="attempts/:aid" element={<Attempt />} />
            <Route path="bookmarks" element={<Bookmarks />} />
            <Route path="my-notes" element={<MyNotes />} />
            <Route path="settings" element={<Settings />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </ToastProvider></AuthProvider>
  </BrowserRouter>
</StrictMode>);
