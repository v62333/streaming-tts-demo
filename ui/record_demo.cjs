// 自動錄 demo 影片：Playwright 操作網頁並錄畫面（無聲），
// 再把伺服器記下的「實際送給喇叭的聲音」依時間戳對齊合成進去。
// 用法：PW=<playwright 模組路徑> node ui/record_demo.cjs   （伺服器需先在 8765 跑著）
const { chromium } = require(process.env.PW || "playwright");
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const BASE = "http://127.0.0.1:8765";
const OUT = path.resolve(__dirname, "..", "results", "video");
fs.mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1,
    recordVideo: { dir: OUT, size: { width: 1280, height: 800 } },
  });
  const page = await ctx.newPage();
  const tVideo0 = Date.now() / 1000;          // 影片時間軸起點（近似）
  const runs = [];
  page.on("response", async (r) => {
    if (r.url().endsWith("/api/run") && r.request().method() === "POST") {
      try { runs.push((await r.json()).run); } catch (_) {}
    }
  });

  await page.goto(BASE);
  await sleep(1500);
  const click = async (sel) => { await page.click(sel); await sleep(600); };
  const waitIdle = () => page.waitForFunction(() => !document.getElementById("play").disabled, null, { timeout: 120000 });

  // 場景 1：FP32，同一段文字對照「整句 vs 逐句」
  await click('#presets button:nth-child(3)');          // 改道
  await click('#variant button[data-v="fp32"]');
  await click('#steps button[data-v="8"]');
  await sleep(800);
  await page.click("#compare");
  // A 跑完、B 開始時把畫面捲到 B
  await page.waitForFunction(() => document.querySelectorAll("#panels .card").length >= 2, null, { timeout: 120000 });
  await page.evaluate(() => document.querySelectorAll("#panels .card")[1]
    .scrollIntoView({ behavior: "smooth", block: "end" }));
  await waitIdle();
  await sleep(800);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await sleep(3500);                                     // 停在結論那行
  await page.evaluate(() => window.scrollTo({ top: 330, behavior: "smooth" }));
  await sleep(3000);

  // 場景 2：INT8 逐句串流，中途暫停 3 秒（看緩衝被填滿＝背壓），再繼續
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await sleep(1000);
  await click('#variant button[data-v="int8"]');
  await click('#mode button[data-v="sentence"]');
  await page.click("#play");
  await sleep(1200);
  await page.evaluate(() => window.scrollTo({ top: 230, behavior: "smooth" }));
  await sleep(2500);
  await page.click("#pause");
  await sleep(3500);
  await page.click("#pause");          // 繼續
  await waitIdle();
  await sleep(3000);

  const video = page.video();
  await ctx.close();
  const webm = await video.path();
  await browser.close();

  // 取回每次播放的聲音與時間戳，交給 mux_video.py 自動對齊合成
  const meta = { webm, t_video0: tVideo0, runs: [] };
  for (const run of runs) {
    const log = await (await fetch(`${BASE}/api/runlog?run=${run}`)).json();
    const ev = log.events.find((e) => e.type === "run");
    const fa = log.events.find((e) => e.type === "first_audio");
    const wav = path.join(OUT, `run_${run}.wav`);
    fs.writeFileSync(wav, Buffer.from(await (await fetch(`${BASE}/api/audio?run=${run}`)).arrayBuffer()));
    meta.runs.push({ run, mode: ev.mode, variant: ev.variant, wav, epoch: ev.epoch,
                     audio_offset_s: log.audio_offset_s, first_audio_t: fa ? fa.t : null });
  }
  const metaPath = path.join(OUT, "runs.json");
  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
  execFileSync(path.resolve(__dirname, "..", ".venv", "bin", "python"),
               [path.join(__dirname, "mux_video.py"), metaPath], { stdio: "inherit" });
})();
