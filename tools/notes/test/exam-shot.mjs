// Sign up, start a paper in practice mode and screenshot its questions (to check how prompts render in the exam hall).
// node exam-shot.mjs OUT http://127.0.0.1:4300 <paper-id> [question numbers to visit, e.g. 1 5 9]
import puppeteer from "puppeteer-core";
const [OUT, B, paper, ...nums] = process.argv.slice(2);
const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH, headless: "new",
  args: ["--no-first-run", ...(process.env.CHROME_ARGS || "").split(" ").filter(Boolean), `--user-data-dir=${OUT}/prof-exam-${Date.now()}`] });
const p = await b.newPage(); await p.setViewport({ width: 1300, height: 900 });
const errors = []; p.on("pageerror", e => errors.push(e.message)); p.on("console", m => m.type() === "error" && errors.push(m.text()));
const sleep = ms => new Promise(r => setTimeout(r, ms));
await p.goto(B + "/signup", { waitUntil: "networkidle0" });
await p.type("#name", "Exam Check"); await p.type("#email", `exam-${Date.now()}@example.com`); await p.type("#password", "a-strong-pass-1");
await Promise.all([p.waitForNavigation({ waitUntil: "networkidle0" }), p.click("button.btn.primary")]);
const r = await p.evaluate(async id => (await fetch(`/api/papers/${id}/start`, { method: "POST", headers: { "content-type": "application/json", "X-Studyloom": "1" }, body: JSON.stringify({ mode: "practice", fresh: true }) })).json(), paper);
await p.goto(`${B}/app/attempts/${r.attempt.id}`, { waitUntil: "networkidle0" }); await sleep(1500);
for (const n of (nums.length ? nums : ["1"])) {
  const btns = await p.$$(".pal-grid button");
  if (btns[Number(n) - 1]) { await btns[Number(n) - 1].click(); await sleep(800); }
  const raw = await p.evaluate(() => { const el = document.querySelector(".q-html"); return el ? { text: el.innerText.slice(0, 160), katex: el.querySelectorAll(".katex").length, rawTex: /\\[a-z]+|\$/.test(el.innerText) } : null; });
  console.log(`Q${n}:`, JSON.stringify(raw));
  await p.screenshot({ path: `${OUT}/exam-q${n}.png` });
}
console.log(errors.length ? "ERRORS: " + errors.join(" | ") : "no page errors");
await b.close();
