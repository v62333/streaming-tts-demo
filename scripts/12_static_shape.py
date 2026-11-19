"""實驗 12：固定輸入長度（不足的補齊）讓加速器的嚴格格式建得起來。

實驗 11 發現 CoreML 的 MLProgram 格式遇到「長度不固定」的輸入會建模型失敗
（推算形狀時卷積輸出太小）。車用 NPU、TensorRT 常見的解法：
  事先把模型的輸入長度固定成一個長度（這裡文字 128、潛在 128，涵蓋測試句最長 102／92），
  每句話補零到這個長度、用遮罩標出哪些是補的，跑完再裁回原長度。

分兩步驗證：
  1. 補齊本身會不會改變結果（全在 CPU 上：補齊 vs 不補齊）
  2. 固定長度的模型交給 CoreML MLProgram：建不建得起來、速度、與 CPU 的差異、錯字率

  python scripts/12_static_shape.py
"""
from __future__ import annotations

import csv
import os
import statistics as S
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))
import supertonic_helper as H  # noqa: E402
from engine import ONNX_DIR, Engine  # noqa: E402
import quality  # noqa: E402

L_BUCKET, T_BUCKET = 128, 128
FIXED = os.path.join(ROOT, "models", "fixed", f"vector_estimator.L{L_BUCKET}_T{T_BUCKET}.onnx")
OUT = os.path.join(ROOT, "results", "12_static_shape.csv")


def make_fixed():
    if os.path.exists(FIXED):
        return
    os.makedirs(os.path.dirname(FIXED), exist_ok=True)
    m = onnx.load(os.path.join(ONNX_DIR, "vector_estimator.onnx"))
    make_dim_param_fixed(m.graph, "batch_size", 1)
    make_dim_param_fixed(m.graph, "text_length", L_BUCKET)
    make_dim_param_fixed(m.graph, "latent_length", T_BUCKET)
    onnx.save(m, FIXED)


def plan_padded(eng: Engine, ve, text, steps=8, speed=1.05, seed=0):
    """同 Engine.plan，但逐步生成那段用補齊到固定長度的模型。"""
    ids, mask = eng.tp([text], ["en"])
    dur, *_ = eng.sess["duration_predictor"].run(None, {"text_ids": ids, "style_dp": eng.style.dp, "text_mask": mask})
    dur = dur / speed
    emb, *_ = eng.sess["text_encoder"].run(None, {"text_ids": ids, "style_ttl": eng.style.ttl, "text_mask": mask})
    rng = np.random.default_rng(seed)
    wav_len = int(dur[0] * eng.sr)
    T = (wav_len + eng.samples_per_latent - 1) // eng.samples_per_latent
    xt = rng.standard_normal((1, eng.ldim * eng.ccf, T)).astype(np.float32)   # 與 Engine.plan 同樣的亂數
    lmask = H.get_latent_mask(np.array([wav_len]), eng.base_chunk, eng.ccf)
    xt = xt * lmask
    L = ids.shape[1]
    assert L <= L_BUCKET and T <= T_BUCKET, (L, T)
    pad = lambda a, n: np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])])  # noqa: E731
    emb_p, mask_p, xt_p, lm_p = pad(emb, L_BUCKET), pad(mask, L_BUCKET), pad(xt, T_BUCKET), pad(lmask, T_BUCKET)
    tot = np.array([steps], dtype=np.float32)
    t0 = time.perf_counter()
    for s in range(steps):
        xt_p, *_ = ve.run(None, {"noisy_latent": xt_p, "text_emb": emb_p, "style_ttl": eng.style.ttl,
                                 "text_mask": mask_p, "latent_mask": lm_p,
                                 "current_step": np.array([s], dtype=np.float32), "total_step": tot})
    ve_s = time.perf_counter() - t0
    wav = eng.vocode(xt_p[:, :, :T])[:wav_len]
    return wav, ve_s


def corpus_wer(pairs):
    return round(100 * sum(e for e, _ in pairs) / sum(n for _, n in pairs), 1)


def main():
    make_fixed()
    eng = Engine()
    ref = {t: eng.synth_full(t, steps=8)[0] for t in quality.SENTS}     # CPU、不補齊＝對照組
    ref_ve = {t: eng.plan(t, steps=8).timings["vector_estimator"] for t in quality.SENTS}
    cfgs = {
        "cpu_padded": ["CPUExecutionProvider"],
        "mlprogram_all": [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}), "CPUExecutionProvider"],
        "mlprogram_ane": [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndNeuralEngine"}), "CPUExecutionProvider"],
        "mlprogram_gpu": [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}), "CPUExecutionProvider"],
    }
    rows = [{"config": "cpu_dynamic(ref)", "load_s": "", "ve_s_median": round(S.median(ref_ve.values()), 3),
             "mel_vs_ref_mean": 0.0, "wer_pct": corpus_wer([quality.wer(t, quality.transcribe(ref[t], eng.sr)) for t in quality.SENTS])}]
    print(rows[-1], flush=True)
    for name, prov in cfgs.items():
        t0 = time.perf_counter()
        try:
            ve = ort.InferenceSession(FIXED, providers=prov)
        except Exception as e:  # noqa: BLE001
            rows.append({"config": name, "error": str(e)[-200:]}); print(rows[-1], flush=True); continue
        load = time.perf_counter() - t0
        plan_padded(eng, ve, quality.SENTS[0])                        # 暖機（首次編譯）
        ves, mels, wers = [], [], []
        for t in quality.SENTS:
            wav, ve_s = plan_padded(eng, ve, t)
            ves.append(ve_s)
            n = min(len(wav), len(ref[t]))
            mels.append(quality.mel_dist(ref[t][:n], wav[:n], eng.sr))
            wers.append(quality.wer(t, quality.transcribe(wav, eng.sr)))
        rows.append({"config": name, "load_s": round(load, 2), "ve_s_median": round(S.median(ves), 3),
                     "mel_vs_ref_mean": round(S.mean(mels), 3), "wer_pct": corpus_wer(wers)})
        print(rows[-1], flush=True)
    keys = ["config", "load_s", "ve_s_median", "mel_vs_ref_mean", "wer_pct", "error"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
