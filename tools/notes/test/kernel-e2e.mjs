// Every runnable cell of the given notes, run in order on the live kernel of a running app, exactly as the reader sends them.
// No browser: signs up over HTTP, reads each note's HTML, posts its cells to /api/kernel/run and reads the NDJSON stream.
// Usage: node tools/notes/test/kernel-e2e.mjs [base-url] [subject-or-note ...]
//        node tools/notes/test/kernel-e2e.mjs http://127.0.0.1:4300 d1-neural-network-foundations
const args = process.argv.slice(2);
const B = args.find(a => /^https?:/.test(a)) || "http://127.0.0.1:4300";
const want = args.filter(a => !/^https?:/.test(a));
let cookie = "";
const api = async (path, body) => {
  const r = await fetch(B + path, { method: body ? "POST" : "GET", headers: { "Content-Type": "application/json", "X-Studyloom": "1", ...(cookie ? { Cookie: cookie } : {}) }, body: body && JSON.stringify(body) });
  const set = r.headers.get("set-cookie"); if (set) cookie = set.split(";")[0];
  return r;
};
const unesc = s => s.replace(/<[^>]+>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&amp;/g, "&");

await api("/api/auth/signup", { name: "Kernel Check", email: `kernel-${Date.now()}@example.com`, password: "a-strong-pass-1" });
const { notes } = await (await api("/api/catalogue")).json();
const pick = notes.filter(n => !want.length || want.some(w => n.id.startsWith(w)));
let failed = 0;
for (const n of pick) {
  const html = await (await fetch(`${B}/content/${n.path}`)).text();
  const cells = [...html.matchAll(/<div class="codewrap pyc[^"]*"[^>]*><pre class="code" data-lang="python">([\s\S]*?)<\/pre>/g)].map(m => unesc(m[1]));
  const t0 = Date.now(); const res = [];
  for (const [i, code] of cells.entries()) {
    const r = await api("/api/kernel/run", { note: n.id, cell: i, code });
    if (r.status !== 200) { res.push(`#${i}: HTTP ${r.status} ${(await r.text()).slice(0, 120)}`); failed++; break; }
    const evs = (await r.text()).trim().split("\n").map(l => JSON.parse(l));
    const done = evs.at(-1), pngs = evs.filter(e => e.t === "png").length, err = evs.find(e => e.t === "err");
    if (!done.ok) { res.push(`#${i}: FAILED ${err ? err.d.trim().split("\n").at(-1) : ""}`); failed++; }
    else res.push(`#${i}: ok${pngs ? ` (${pngs} png)` : ""}`);
  }
  await api("/api/kernel/restart", { note: n.id });
  console.log(`${n.id}: ${cells.length} cells in ${((Date.now() - t0) / 1000).toFixed(1)} s\n  ${res.join("  ")}`);
}
console.log(failed ? `${failed} cell(s) failed` : "every cell ran on the live kernel");
process.exit(failed ? 1 : 0);
