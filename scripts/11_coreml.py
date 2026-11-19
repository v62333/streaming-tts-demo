"""實驗 11：換成硬體加速器（Apple CoreML：可用 GPU／神經網路引擎）會怎樣。

跟 TensorRT、車用 NPU 是同一類問題：哪些運算加速器吃得下、模型被切成幾塊、
輸入長度不固定時能不能建起來、精度有沒有變、第一次要不要先編譯。

每個設定開獨立行程，只換「哪個模型交給 CoreML」，其餘留在 CPU：
  量：建立 session 時間、第一句（含 CoreML 首次編譯）、穩態每句時間（同一句 3 次中位數）、
     換一句不同長度的第一次（看長度改變會不會重新編譯）、
     與 CPU FP32 同種子輸出的頻譜差（基準：只換隨機種子＝1.525）

  python scripts/11_coreml.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics as S
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "11_coreml.csv")

CHILD = r'''
import json, os, statistics as S, sys, time
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.join(sys.argv[1], "src"))
from engine import Engine, ONNX_DIR
from quality import mel_dist
cfg = json.loads(sys.argv[2])
A = "Traffic is heavy on the route ahead, so I found a faster way that saves about twelve minutes."
B = "Your battery is at twenty percent. The nearest charging station is four kilometers ahead."
ov = {"vector_estimator": os.path.join(sys.argv[1], "models/int8/vector_estimator.int8.onnx")} if cfg.get("int8") else {}
eng = Engine(model_overrides=ov)
ref = Engine()                                   # CPU FP32 對照（同種子）
load = {}
for name, prov in cfg.get("coreml", {}).items():
    path = ov.get(name, os.path.join(ONNX_DIR, f"{name}.onnx"))
    t0 = time.perf_counter()
    eng.sess[name] = ort.InferenceSession(path, providers=[("CoreMLExecutionProvider", prov), "CPUExecutionProvider"])
    load[name] = round(time.perf_counter() - t0, 2)
def run(text):
    t0 = time.perf_counter(); wav, t = eng.synth_full(text, steps=8); return time.perf_counter() - t0, wav, t
first, wavA, _ = run(A)
steady = []
for _ in range(3):
    dt, wavA, tA = run(A); steady.append(dt)
firstB, wavB, _ = run(B)
refA, _ = ref.synth_full(A, steps=8)
n = min(len(refA), len(wavA))
print(json.dumps({"load_s": load, "first_s": round(first, 2), "steady_s": round(S.median(steady), 3),
                  "ve_s": round(tA["vector_estimator"], 3), "voc_s": round(tA["vocoder"], 3),
                  "first_new_len_s": round(firstB, 2), "audio_s": round(len(wavA) / eng.sr, 2),
                  "mel_vs_cpu": round(float(mel_dist(refA[:n], wavA[:n], eng.sr)), 3)}))
'''

NN = {"ModelFormat": "NeuralNetwork", "MLComputeUnits": "ALL"}
CONFIGS = {
    "cpu_fp32": {},
    "cpu_int8ve": {"int8": True},
    "ve_nn_all": {"coreml": {"vector_estimator": NN}},
    "ve_nn_gpu": {"coreml": {"vector_estimator": {**NN, "MLComputeUnits": "CPUAndGPU"}}},
    "ve_nn_ane": {"coreml": {"vector_estimator": {**NN, "MLComputeUnits": "CPUAndNeuralEngine"}}},
    "voc_mlprogram": {"coreml": {"vocoder": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}}},
    "voc_nn": {"coreml": {"vocoder": NN}},
    "all4_nn": {"coreml": {k: NN for k in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")}},
    "int8ve_on_coreml": {"int8": True, "coreml": {"vector_estimator": NN}},
}


def main():
    only = sys.argv[1:] or list(CONFIGS)
    rows = []
    for name in only:
        r = subprocess.run([sys.executable, "-c", CHILD, ROOT, json.dumps(CONFIGS[name])],
                           capture_output=True, text=True, cwd=ROOT)
        js = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode or not js:
            err = [l for l in r.stderr.splitlines() if "rror" in l or "FAIL" in l][-1:] or [r.stderr[-300:]]
            d = {"config": name, "error": f"rc={r.returncode} " + err[0][-300:]}
        else:
            d = {"config": name, **json.loads(js[-1])}
        rows.append(d)
        print(json.dumps(d, ensure_ascii=False), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "config", k))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
