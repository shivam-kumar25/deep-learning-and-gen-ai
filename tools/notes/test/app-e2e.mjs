// Studyloom (DL) end to end: landing → sign up → library → subject → reader: run a code cell on the live Python kernel
// (the cells above it run first), add a scratch cell and run PyTorch in it, restart the kernel → mark a section understood → desk.
// Usage: node app-e2e.mjs [base-url]   (default http://127.0.0.1:4300); screenshots go to ./shots (or $SHOTS)
import puppeteer from "puppeteer-core";
import fs from "node:fs";
const B = process.argv.slice(2).find(a => /^https?:/.test(a)) || "http://127.0.0.1:4300", OUT = process.env.SHOTS || new URL("./shots/", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
fs.mkdirSync(OUT, { recursive: true });
const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: "new",
  args: ["--no-first-run", ...(process.env.CHROME_ARGS || "").split(" ").filter(Boolean), `--user-data-dir=${OUT}/profile-${Date.now()}`] });
const p = await b.newPage();
await p.setViewport({ width: 1440, height: 900 });
const errors = [];
p.on("pageerror", e => errors.push("pageerror: " + e.message));
p.on("console", m => { if (m.type() === "error" && !/favicon/.test(m.text())) errors.push("console: " + m.text()); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const shot = async (n, full = false) => { await sleep(600); await p.screenshot({ path: `${OUT}/${n}.png`, fullPage: full }); console.log("shot", n); };
const reader = () => p.frames().find(fr => fr.url().includes("/content/"));
const waitOut = async (f, cell, re, ms = 120000) => {   // wait until the cell's live output matches re (or shows an error)
  const t0 = Date.now();
  for (;;) {
    const txt = await cell.evaluate(el => el.querySelector(".outblk.live .rows")?.innerText || "");
    if (re.test(txt) || /Error|refused|could not/i.test(txt) || Date.now() - t0 > ms) return txt;
    await sleep(300);
  }
};

await p.goto(B + "/", { waitUntil: "networkidle0" }); await shot("01-landing", true);
await p.goto(B + "/signup", { waitUntil: "networkidle0" });
await p.type("#name", "DL Learner"); await p.type("#email", `dl-${Date.now()}@example.com`); await p.type("#password", "a-strong-pass-1");
await Promise.all([p.waitForNavigation({ waitUntil: "networkidle0" }), p.click("button.btn.primary")]);
await shot("02-desk");
await p.goto(B + "/app/library", { waitUntil: "networkidle0" }); await shot("03-library", true);
console.log("subjects in library:", await p.$$eval("a[href^='/app/s/']", e => new Set(e.map(a => a.getAttribute("href"))).size));
console.log("library tracks:", JSON.stringify(await p.$$eval(".group-h h2", e => e.map(h => h.textContent))));
await p.goto(B + "/app/s/d1-neural-network-foundations", { waitUntil: "networkidle0" }); await shot("04-subject");
console.log("notes in D1:", await p.$$eval("a.note-row", e => e.length));

// note 02: run the "visualise all three" cell; the perceptron cell above it runs first, on the kernel
await p.goto(B + "/app/learn/d1-neural-network-foundations/02_neuron_and_perceptron", { waitUntil: "networkidle0" }); await sleep(2500);
{
  const f = reader();
  const cells = await f.$$(".codewrap.pyc");
  console.log("runnable cells in note 02:", cells.length, "| maths typeset:", await f.$$eval(".katex", e => e.length));
  const target = cells.at(-1);
  await target.evaluate(el => el.scrollIntoView({ block: "center" }));
  await (await target.$("button.run")).click();
  const out = await waitOut(f, target, /plot|img/i, 120000);
  const imgs = await target.$$eval(".outblk.live img", e => e.length);
  console.log("live run → images:", imgs, "| text:", JSON.stringify(out.slice(0, 120)));
  console.log("kernel chip:", await f.$eval(".kchip .kt", el => el.textContent));
  await shot("05-live-run");
  // the perceptron cell (run first, invisibly) set `X`; a scratch cell sees it, and runs torch
  const add = await target.evaluateHandle(el => [...el.querySelectorAll(".codehead button")].find(x => x.textContent === "+ cell"));
  await add.click(); await sleep(400);
  const scratch = await f.$(".codewrap.scratch");
  const ta = await scratch.$("textarea.pyedit");
  await ta.evaluate(el => { el.value = ""; });
  await ta.type("import torch\nt = torch.tensor(X, dtype=torch.float32)\nprint(t.shape, t.sum().item(), torch.cuda.is_available())");
  await (await scratch.$("button.run")).click();
  console.log("scratch cell:", JSON.stringify((await waitOut(f, scratch, /torch\.Size/)).trim()));
  await shot("06-scratch-torch");
  await (await f.$(".kchip button")).click(); await sleep(800);
  console.log("after restart:", await f.$eval(".kchip .kt", el => el.textContent));
  await f.evaluate(() => document.querySelector(".donebtn").scrollIntoView({ block: "center" }));
  await (await f.$(".donebtn")).click(); await sleep(800);
}
await p.goto(B + "/app", { waitUntil: "networkidle0" }); await shot("07-desk-after");
console.log(errors.length ? "ERRORS:\n" + errors.join("\n") : "no console errors");
await b.close();
