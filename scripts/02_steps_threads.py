"""實驗 2：逐步生成的步數、執行緒數，對整句延遲的影響。"""
import csv, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from engine import Engine
S = {"short": "Turn left.",
     "long": "Your battery is at twenty percent. The nearest charging station is four kilometers ahead, on the right side of the highway."}
rows = []
for th in (0, 4, 1):
    eng = Engine(threads=th)
    eng.synth_full("Warm up.")
    for steps in (8, 4, 2, 1):
        for name, s in S.items():
            ve, tot = [], []
            for _ in range(5):
                t0 = time.perf_counter(); wav, t = eng.synth_full(s, steps=steps); tot.append(time.perf_counter() - t0); ve.append(t["vector_estimator"])
            r = {"threads": th or "auto", "steps": steps, "sent": name, "audio_s": round(len(wav)/eng.sr, 2),
                 "ve_ms": round(np.median(ve)*1000, 1), "per_step_ms": round(np.median(ve)*1000/steps, 1),
                 "total_ms": round(np.median(tot)*1000, 1), "p_max_ms": round(max(tot)*1000, 1)}
            rows.append(r); print(r, flush=True)
os.makedirs("results", exist_ok=True)
with open("results/02_steps_threads.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
