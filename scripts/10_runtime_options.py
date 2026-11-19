"""實驗 10：冷啟動時間與「量化以外」的記憶體調校。

每個設定開獨立行程量 3 次（避免前一次的記憶體與快取互相污染），量：
  載入：建立 4 個推論 session 的時間（車子發動到「模型準備好」）
  第一句：載入後第一次合成的時間（有些初始化是第一次跑才做）
  之後：第 2～4 次合成的中位數（穩態）
  記憶體：載入後增加多少、整個行程峰值

設定（都以 INT8 逐步生成版本為主，FP32 預設當參考）：
  default         ORT 預設（載入時做完整圖優化、開記憶體池、開記憶體配置規劃）
  no_graph_opt    載入時完全不做圖優化（看圖優化本身值多少）
  preopt          事先把圖優化好存成檔，載入時只剩少量優化（省載入時間）
  no_arena        關掉 CPU 記憶體池（用完就還，不預先保留）
  no_mem_pattern  關掉記憶體配置規劃（不依第一次執行的形狀預先排好緩衝區）
  lean            no_arena + no_mem_pattern

注意：作業系統會快取剛讀過的模型檔，所以這裡量的是「檔案已在記憶體快取」的載入時間；
真車冷開機要從快閃記憶體讀檔，會再多一段讀檔時間（另外量檔案大小）。

  python scripts/10_runtime_options.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics as S
import subprocess
import sys

import onnxruntime as ort

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from engine import ONNX_DIR  # noqa: E402

OUT = os.path.join(ROOT, "results", "10_runtime_options.csv")
INT8_VE = os.path.join(ROOT, "models", "int8", "vector_estimator.int8.onnx")
OPT_DIR = os.path.join(ROOT, "models", "preopt")
NAMES = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
TEXT = ("Traffic is heavy on the route ahead, so I found a faster way that saves about twelve minutes. "
        "It goes through the city center, and there is one toll road.")

CHILD = r'''
import json, os, resource, statistics as S, sys, time
import psutil
sys.path.insert(0, os.path.join(sys.argv[1], "src"))
import onnxruntime as ort
from engine import Engine
cfg = json.loads(sys.argv[2])
sc = {}
for k, v in cfg.get("session_config", {}).items():
    sc[k] = getattr(ort.GraphOptimizationLevel, v) if k == "graph_optimization_level" else v
p = psutil.Process()
base = p.memory_info().rss
t0 = time.perf_counter()
eng = Engine(model_overrides=cfg.get("overrides"), session_config=sc)
load = time.perf_counter() - t0
loaded = p.memory_info().rss
text = sys.argv[3]
t0 = time.perf_counter(); eng.synth_full(text, steps=8); first = time.perf_counter() - t0
rest = []
for _ in range(3):
    t0 = time.perf_counter(); eng.synth_full(text, steps=8); rest.append(time.perf_counter() - t0)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS: bytes
print(json.dumps({"load_s": round(load, 3), "load_ve_s": round(eng.load_s["vector_estimator"], 3),
                  "first_synth_s": round(first, 3), "steady_synth_s": round(S.median(rest), 3),
                  "after_load_MB": round((loaded - base) / 2**20), "peak_rss_MB": round(peak / 2**20)}))
'''


def make_preopt():
    """事先存一份圖優化好的模型（EXTENDED：與硬體無關的優化，可攜）。"""
    os.makedirs(OPT_DIR, exist_ok=True)
    src = {k: os.path.join(ONNX_DIR, f"{k}.onnx") for k in NAMES}
    src["vector_estimator"] = INT8_VE
    out = {}
    for k, path in src.items():
        dst = os.path.join(OPT_DIR, f"{k}.onnx")
        if not os.path.exists(dst):
            o = ort.SessionOptions()
            o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            o.optimized_model_filepath = dst
            ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])
        out[k] = dst
    return out


def main():
    preopt = make_preopt()
    int8 = {"vector_estimator": INT8_VE}
    configs = {
        "fp32_default": {},
        "default": {"overrides": int8},
        "no_graph_opt": {"overrides": int8, "session_config": {"graph_optimization_level": "ORT_DISABLE_ALL"}},
        "preopt": {"overrides": preopt},
        "no_arena": {"overrides": int8, "session_config": {"enable_cpu_mem_arena": False}},
        "no_mem_pattern": {"overrides": int8, "session_config": {"enable_mem_pattern": False}},
        "lean": {"overrides": int8, "session_config": {"enable_cpu_mem_arena": False, "enable_mem_pattern": False}},
    }
    # 暖機一次（讓模型檔進作業系統快取，各設定的起跑點一致），不計
    subprocess.run([sys.executable, "-c", CHILD, ROOT, json.dumps(configs["default"]), TEXT],
                   capture_output=True, text=True, check=True)
    rows = []
    for rep in range(3):              # 交錯執行，避免某個設定剛好碰上背景尖峰
        for name, cfg in configs.items():
            r = subprocess.run([sys.executable, "-c", CHILD, ROOT, json.dumps(cfg), TEXT],
                               capture_output=True, text=True)
            if r.returncode:
                print(name, "FAILED", r.stderr[-500:]); continue
            d = {"config": name, "rep": rep, **json.loads(r.stdout.strip().splitlines()[-1])}
            rows.append(d)
            print(json.dumps(d), flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\n設定             載入s  第一句s  穩態s  載入後MB  峰值MB")
    for name in configs:
        g = [r for r in rows if r["config"] == name]
        m = lambda k: S.median(r[k] for r in g)  # noqa: E731
        print(f"{name:15s} {m('load_s'):6.2f} {m('first_synth_s'):7.2f} {m('steady_synth_s'):6.2f}"
              f" {m('after_load_MB'):8.0f} {m('peak_rss_MB'):7.0f}")
    sizes = {k: round(os.path.getsize(v) / 2**20) for k, v in preopt.items()}
    print("preopt 檔案大小 MB:", sizes)


if __name__ == "__main__":
    main()
