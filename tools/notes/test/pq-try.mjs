// Answer practice questions on a page and screenshot each one with its feedback and solution.
// node pq-try.mjs OUT page.html "N=answer" ...   N = question number on the page (1-based); answer = "A,C" or a number
import puppeteer from "puppeteer-core";
const [OUT, page, ...tries] = process.argv.slice(2);
const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: "new",
  args: ["--no-first-run", ...(process.env.CHROME_ARGS || "").split(" ").filter(Boolean), `--user-data-dir=${OUT}/prof-pq-${Date.now()}`] });
const p = await b.newPage(); await p.setViewport({ width: 1300, height: 900, deviceScaleFactor: 1 });
const errors = []; p.on("pageerror", e => errors.push(e.message));
await p.goto("http://127.0.0.1:8765/" + page, { waitUntil: "networkidle0" });
const qs = await p.$$(".pq");
for (const t of tries) {
  const [n, ans] = t.split("=");
  const q = qs[Number(n) - 1];
  await q.evaluate(el => el.scrollIntoView({ block: "start" }));
  const num = await q.$(".pq-num input");
  if (num) { await num.type(ans); await (await q.$(".pq-check")).click(); }
  else {
    for (const k of ans.split(",")) await (await q.$(`li[data-k="${k.trim()}"]`)).click();
    const c = await q.$(".pq-check"); if (c && await c.evaluate(e => !e.hidden)) await c.click();
  }
  await new Promise(r => setTimeout(r, 500));
  console.log(`Q${n} answered ${ans}:`, await q.$eval(".pq-head", e => e.innerText.replace(/\s+/g, " ")));
  await q.screenshot({ path: `${OUT}/pq-${n}.png` });
}
console.log(errors.length ? "ERRORS: " + errors.join(" | ") : "no page errors");
await b.close();
