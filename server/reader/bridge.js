/* Studyloom reader bridge: injected at the top of every note page served under /content.
   The note runs inside the reader's iframe on the same origin, so it talks to the app through
   window.parent.__studyloom (set up by the Reader before the page loads):
     theme()            -> "light" | "dark"        applied before notes.js picks a theme
     hue                -> the subject's colour     used by reader.css
     done()             -> [section ids]            seeded into the key notes.js reads, so its buttons show your server-side progress
     onDone(ids)        <- every time a "Mark as understood" button changes the list
     onSection(id)      <- the section you are reading
     onNavigate(path)   <- a click on a link to another note; return true to let the app route it
   Opened on its own (not in the reader), the page behaves exactly as before. */
(function () {
  var host;
  try { host = window.parent !== window && window.parent.__studyloom; } catch (e) { host = null; }
  if (!host) return;
  var root = document.documentElement;
  root.classList.add("in-studyloom");
  try {
    var t = host.theme();
    root.dataset.theme = t;
    localStorage.setItem("ml-notes-theme", JSON.stringify(t));
  } catch (e) {}
  if (host.hue != null) root.style.setProperty("--sl-h", host.hue);

  window.__slTheme = function (t) { root.dataset.theme = t; try { localStorage.setItem("ml-notes-theme", JSON.stringify(t)); } catch (e) {} };
  window.__slGoto = function (id) {
    var el = id && document.getElementById(id), m = /^pq-(\d+)$/.exec(id || "");
    if (!el && m) el = document.querySelectorAll(".pq")[Number(m[1]) - 1];   // a question, by its number on the page
    if (el) el.scrollIntoView({ behavior: "smooth", block: m ? "center" : "start" }); else window.scrollTo({ top: 0, behavior: "smooth" });
    if (el && m) { el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
  };

  document.addEventListener("DOMContentLoaded", function () {
    var b = document.body;
    var key = "ml-" + (b.dataset.subject || "ml00") + "-" + b.dataset.page + "-done";
    try { localStorage.setItem(key, JSON.stringify(host.done() || [])); } catch (e) {}
    var setItem = Storage.prototype.setItem;
    Storage.prototype.setItem = function (k, v) {
      setItem.apply(this, arguments);
      if (k === key) { try { host.onDone(JSON.parse(v)); } catch (e) {} }
    };

    var current = null, timer = null;
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting || en.target.id === current) return;
        current = en.target.id;
        clearTimeout(timer);
        timer = setTimeout(function () { try { host.onSection(current); } catch (e) {} }, 250);
      });
    }, { rootMargin: "-20% 0px -65% 0px" });
    document.querySelectorAll("section.sec[id]").forEach(function (s) { spy.observe(s); });
  }, { once: true });

  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest("a[href]");
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || a.target === "_blank") return;
    var url;
    try { url = new URL(a.getAttribute("href"), location.href); } catch (err) { return; }
    if (url.origin !== location.origin) { a.target = "_blank"; a.rel = "noopener"; return; }
    if (url.pathname === location.pathname) return;                 // an anchor on this page
    try { if (host.onNavigate(url.pathname, url.hash.slice(1))) e.preventDefault(); } catch (err) {}
  }, true);
})();
