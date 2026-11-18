"""實驗 1：整句生成時，時間花在哪個階段？隨句長怎麼變？"""
import csv
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from engine import Engine  # noqa: E402

SENTENCES = [
    "Turn left.",
    "In two hundred meters, turn right onto Main Street.",
    "Your battery is at twenty percent. The nearest charging station is four kilometers ahead, on the right side of the highway.",
    "Traffic is heavy on the route ahead, so I found a faster way that saves about twelve minutes. It goes through the city center, and there is one toll road. Would you like me to switch to the new route, or keep the current one?",
]
REPEAT = 5
STAGES = ["text_proc", "duration", "text_encoder", "vector_estimator", "vocoder"]

out = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out, exist_ok=True)
eng = Engine()
eng.synth_full("Warm up.")  # 暖機，不計

rows = []
for i, s in enumerate(SENTENCES):
    acc = {k: [] for k in STAGES + ["total"]}
    for r in range(REPEAT):
        t0 = time.perf_counter()
        wav, t = eng.synth_full(s)
        tot = time.perf_counter() - t0
        for k in STAGES:
            acc[k].append(t[k])
        acc["total"].append(tot)
    audio_s = len(wav) / eng.sr
    med = {k: float(np.median(v)) for k, v in acc.items()}
    rows.append({"id": i, "chars": len(s), "audio_s": round(audio_s, 2),
                 **{k: round(med[k] * 1000, 1) for k in med},
                 "rtf": round(med["total"] / audio_s, 3)})
    sf.write(os.path.join(out, f"01_full_{i}.wav"), wav, eng.sr)

with open(os.path.join(out, "01_stage_profile.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("單位 ms（中位數，n=%d）" % REPEAT)
for r in rows:
    print(r)
