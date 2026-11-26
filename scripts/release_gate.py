"""發版品質關卡：候選版本要過所有檢查才准發版，否則結束碼 1（CI 會亮紅燈）。

  python scripts/release_gate.py int8_ve           # 預期通過
  python scripts/release_gate.py int8_all          # 預期擋下（聲碼器量化壞掉）

每一項都跟「同一台機器上現場量的 baseline」比，所以門檻可以跨硬體共用。
量測項目：
  1. 錄音辨識錯字率（whisper 轉回文字）     → 念錯、念糊
  2. 頻譜偏離（相對 baseline、同隨機種子）     → 音色跑掉；門檻＝「只換隨機種子」的自然變化量
  3. 生成速度（生成耗時 ÷ 聲音長度）          → 不准比 baseline 慢
  4. 峰值記憶體（獨立行程量）                 → 不准比 baseline 多 5% 以上
  5. 串流首段延遲（逐句串流、模擬音效卡節拍）   → 不准比 baseline 慢 10% 以上
  6. 串流斷音次數                             → 必須 0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from engine import Engine  # noqa: E402
from quality import SENTS, mel_dist, transcribe, wer  # noqa: E402
from streamer import stream  # noqa: E402

STREAM_TEXTS = [
    "In two hundred meters, turn right onto Main Street. Then keep left at the fork. Your destination will be on the right.",
    "Your battery is at twenty percent. The nearest charging station is four kilometers ahead, on the right side of the highway.",
    "New message from Anna. She says she is running late and will meet you at the restaurant at seven thirty. Do you want to reply?",
]

MEM_CODE = r"""
import json, os, sys, resource
sys.path.insert(0, "src")
from engine import Engine
cfg = json.loads(sys.argv[1])
eng = Engine(model_overrides=cfg["overrides"])
eng.synth_full(sys.argv[2], steps=cfg["steps"])
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 1024   # macOS: bytes, Linux: KB
print(json.dumps({"peak_mb": round(peak_mb, 1)}))
"""


def peak_mem(cfg) -> float:
    txt = STREAM_TEXTS[1]
    vals = []
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", MEM_CODE, json.dumps(cfg), txt],
                             capture_output=True, text=True, check=True).stdout
        vals.append(json.loads(out.strip().splitlines()[-1])["peak_mb"])
    return float(np.median(vals))


def measure(name, cfg, ref_wavs=None, with_asr=True, seed_offset=0):
    eng = Engine(model_overrides=cfg["overrides"])
    eng.synth_full("Warm up.", steps=cfg["steps"])
    wavs, gen, aud, errs, words, fails = [], 0.0, 0.0, 0, 0, []
    for i, s in enumerate(SENTS):
        t0 = time.perf_counter()
        wav, _ = eng.synth_full(s, steps=cfg["steps"], seed=i + seed_offset)
        gen += time.perf_counter() - t0
        aud += len(wav) / eng.sr
        wavs.append(wav)
        if with_asr:
            hyp = transcribe(wav, eng.sr)
            e, n = wer(s, hyp)
            errs += e
            words += n
            if e:
                fails.append({"ref": s, "heard": hyp})
    m = {"name": name, "rtf": gen / aud, "wavs": wavs, "sr": eng.sr, "misheard": fails}
    if with_asr:
        m["wer_pct"] = 100 * errs / words
    if ref_wavs is not None:
        m["mel_dist"] = float(np.mean([mel_dist(r, w, eng.sr) for r, w in zip(ref_wavs, wavs)]))
    fa, ur = [], 0
    if with_asr:  # 串流只量正式候選與 baseline
        for t in STREAM_TEXTS:
            st = stream(eng, t, policy="sentence", steps=cfg["steps"], prefill_s=0.2)
            fa.append(st.t_first_audio - st.t_submit)
            ur += st.underruns
        m["first_audio_p50_s"] = float(np.median(fa))
        m["underruns"] = ur
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--out", default="results/gate")
    ap.add_argument("--quality-only", action="store_true",
                    help="只用品質項目（錯字率、頻譜偏離）判定；效能項目照量照報，但不影響結果。"
                         "給 CI 的共用 x86 機器用：效能門檻要在目標硬體上判定。")
    a = ap.parse_args()
    cands = json.load(open("configs/release_candidates.json"))
    th = json.load(open("configs/gate_thresholds.json"))
    if a.candidate not in cands or a.candidate.startswith("_"):
        sys.exit(f"unknown candidate {a.candidate}")
    cfg, base_cfg = cands[a.candidate], cands["baseline"]
    t_start = time.time()

    print("[1/4] baseline …", flush=True)
    base = measure("baseline", base_cfg)
    print("[2/4] 自然變化量（baseline 只換隨機種子）…", flush=True)
    noise = measure("seed_noise", base_cfg, ref_wavs=base["wavs"], with_asr=False, seed_offset=1000)
    print(f"[3/4] 候選 {a.candidate} …", flush=True)
    cand = measure(a.candidate, cfg, ref_wavs=base["wavs"])
    print("[4/4] 峰值記憶體（獨立行程）…", flush=True)
    base_mem, cand_mem = peak_mem(base_cfg), peak_mem(cfg)

    checks = []

    def check(item, value, limit, ok, unit=""):
        kind = "quality" if len(checks) < 3 else "perf"     # 前 3 項是品質，後 4 項是效能
        checks.append({"item": item, "value": value, "limit": limit, "unit": unit, "pass": bool(ok), "kind": kind})

    check("錄音辨識錯字率", round(cand["wer_pct"], 2), f"≤ {th['wer_max_pct']}", cand["wer_pct"] <= th["wer_max_pct"], "%")
    d = cand["wer_pct"] - base["wer_pct"]
    check("錯字率比 baseline 增加", round(d, 2), f"≤ {th['wer_max_delta_pct']}", d <= th["wer_max_delta_pct"], "pt")
    lim = th["mel_dist_max_ratio_to_seed_noise"] * noise["mel_dist"]
    check("頻譜偏離（vs baseline）", round(cand["mel_dist"], 3), f"≤ {lim:.3f}（自然變化量）", cand["mel_dist"] <= lim)
    lim = th["rtf_max_ratio_to_baseline"] * base["rtf"]
    check("生成速度 RTF", round(cand["rtf"], 3), f"≤ {lim:.3f}（baseline）", cand["rtf"] <= lim, "×")
    lim = th["peak_mem_max_ratio_to_baseline"] * base_mem
    check("峰值記憶體", round(cand_mem), f"≤ {lim:.0f}", cand_mem <= lim, "MB")
    lim = th["first_audio_p50_max_ratio_to_baseline"] * base["first_audio_p50_s"]
    check("串流首段延遲 p50", round(cand["first_audio_p50_s"], 3), f"≤ {lim:.3f}", cand["first_audio_p50_s"] <= lim, "s")
    check("串流斷音", cand["underruns"], f"≤ {th['underruns_max']}", cand["underruns"] <= th["underruns_max"], "次")

    judged = [c for c in checks if c["kind"] == "quality"] if a.quality_only else checks
    passed = all(c["pass"] for c in judged)
    mode = "quality-only" if a.quality_only else "full"
    report = {
        "candidate": a.candidate, "config": cfg, "mode": mode, "passed": passed, "checks": checks,
        "baseline": {"wer_pct": round(base["wer_pct"], 2), "rtf": round(base["rtf"], 3),
                     "peak_mb": round(base_mem), "first_audio_p50_s": round(base["first_audio_p50_s"], 3)},
        "speedup_vs_baseline": round(base["rtf"] / cand["rtf"], 2),
        "misheard": cand["misheard"][:5],
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "cpu_count": os.cpu_count()},
        "elapsed_s": round(time.time() - t_start, 1),
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(report, open(f"{a.out}/{a.candidate}.json", "w"), ensure_ascii=False, indent=2)

    title = f"## 發版關卡：`{a.candidate}` → {'✅ 准許發版' if passed else '❌ 擋下'}"
    if a.quality_only:
        title = f"## 發版關卡（只判定品質）：`{a.candidate}` → {'✅ 品質通過' if passed else '❌ 品質擋下'}"
    md = [title, "",
          f"速度是 baseline 的 **{report['speedup_vs_baseline']}×**；機器：{report['machine']['platform']}，"
          f"{report['machine']['cpu_count']} 核；耗時 {report['elapsed_s']} 秒", "",
          "| 檢查 | 結果 | 門檻 | |", "|---|---|---|---|"]
    for c in checks:
        mark = "✅" if c["pass"] else "❌"
        if a.quality_only and c["kind"] == "perf":
            mark += "（參考，不判定）"
        md.append(f"| {c['item']} | {c['value']} {c['unit']} | {c['limit']} {c['unit']} | {mark} |")
    if a.quality_only:
        md += ["", "效能項目只在目標硬體上判定；這台機器的效能數字僅供參考。"]
    if cand["misheard"]:
        md += ["", "聽錯的句子（前 5 句）：", ""]
        md += [f"- 原文「{m['ref']}」→ 聽成「{m['heard']}」" for m in cand["misheard"][:5]]
    md = "\n".join(md) + "\n"
    open(f"{a.out}/{a.candidate}.md", "w").write(md)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(md)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
