"""實驗 3：串流 vs 整句。切段策略 × 步數 × 執行緒，量首段延遲與斷音。

播放端用 SimulatedPlayer（牆鐘節拍，跟音效卡同節奏），所以每次執行耗時約等於聲音長度。
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from engine import Engine  # noqa: E402
from streamer import stream  # noqa: E402

TEXTS = {
    "nav": "In two hundred meters, turn right onto Main Street. Then keep left at the fork. Your destination will be on the right.",
    "ev": "Your battery is at twenty percent. The nearest charging station is four kilometers ahead, on the right side of the highway. Charging to eighty percent will take about thirty minutes.",
    "reroute": "Traffic is heavy on the route ahead, so I found a faster way that saves about twelve minutes. It goes through the city center, and there is one toll road. Would you like me to switch to the new route, or keep the current one?",
    "msg": "New message from Anna. She says she is running late and will meet you at the restaurant at seven thirty. Do you want to reply?",
}

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, nargs="+", default=[0, 1])
ap.add_argument("--steps", type=int, nargs="+", default=[8, 4])
ap.add_argument("--policies", nargs="+", default=["whole", "sentence", "first_clause"])
ap.add_argument("--prefills", type=float, nargs="+", default=[0.2])
ap.add_argument("--variant", default="fp32", choices=["fp32", "int8ve"])
ap.add_argument("--tag", default="native")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--out", default="results/03_streaming_bench.csv")
a = ap.parse_args()

rows = []
for th in a.threads:
    ov = {"vector_estimator": "models/int8/vector_estimator.int8.onnx"} if a.variant == "int8ve" else {}
    eng = Engine(threads=th, model_overrides=ov)
    eng.synth_full("Warm up.")
    for steps in a.steps:
        for pol in a.policies:
          for pf in a.prefills:
            for name, txt in TEXTS.items():
                for r in range(a.repeat):
                    st = stream(eng, txt, policy=pol, steps=steps, prefill_s=pf)
                    row = {"tag": a.tag, "variant": a.variant, "threads": th or "auto",
                           "steps": steps, "policy": pol, "prefill_s": pf,
                           "text": name, "rep": r, **st.summary()}
                    rows.append(row)
                    print(json.dumps(row), flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
