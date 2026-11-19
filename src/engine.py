"""Supertonic-3 推論引擎：把官方整句流程拆成可分開計時、可分段解碼的階段。

官方參考實作：third_party/supertonic_helper.py（MIT, Supertone Inc.）
本檔只重組呼叫順序，不改模型。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party"))
import supertonic_helper as H  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX_DIR = os.path.join(ROOT, "models", "supertonic-3", "onnx")
VOICE_DIR = os.path.join(ROOT, "models", "supertonic-3", "voice_styles")


@dataclass
class Plan:
    """整句前段（文字→潛在表示）的結果，交給聲碼器（潛在表示→聲波）。"""
    latent: np.ndarray          # [1, ldim*ccf, T]
    wav_len: int                # 有效樣本數
    timings: dict = field(default_factory=dict)


class Engine:
    def __init__(self, onnx_dir: str = ONNX_DIR, voice: str = "M1",
                 threads: int = 0, providers=None, model_overrides: dict | None = None,
                 session_config: dict | None = None):
        opts = ort.SessionOptions()
        # 推論引擎設定（記憶體池、圖優化等級…），名稱同 ort.SessionOptions 的屬性
        for k, v in (session_config or {}).items():
            setattr(opts, k, v)
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        providers = providers or ["CPUExecutionProvider"]
        paths = {k: os.path.join(onnx_dir, f"{k}.onnx") for k in
                 ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")}
        paths.update(model_overrides or {})
        self.load_s = {}
        self.sess = {}
        for k, p in paths.items():
            t0 = time.perf_counter()
            self.sess[k] = ort.InferenceSession(p, sess_options=opts, providers=providers)
            self.load_s[k] = time.perf_counter() - t0
        self.cfgs = H.load_cfgs(onnx_dir)
        self.tp = H.load_text_processor(onnx_dir)
        self.style = H.load_voice_style([os.path.join(VOICE_DIR, f"{voice}.json")])
        self.sr = self.cfgs["ae"]["sample_rate"]
        self.base_chunk = self.cfgs["ae"]["base_chunk_size"]
        self.ccf = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]
        # 一個潛在格（latent frame）對應幾個聲音樣本
        self.samples_per_latent = self.base_chunk * self.ccf

    # ---------- 前段：整句一次算（韻律、長度由整句決定） ----------
    def plan(self, text: str, lang: str = "en", steps: int = 8, speed: float = 1.05,
             seed: int | None = 0) -> Plan:
        t = {}
        t0 = time.perf_counter()
        ids, mask = self.tp([text], [lang])
        t["text_proc"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dur, *_ = self.sess["duration_predictor"].run(
            None, {"text_ids": ids, "style_dp": self.style.dp, "text_mask": mask})
        dur = dur / speed
        t["duration"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        emb, *_ = self.sess["text_encoder"].run(
            None, {"text_ids": ids, "style_ttl": self.style.ttl, "text_mask": mask})
        t["text_encoder"] = time.perf_counter() - t0

        rng = np.random.default_rng(seed)
        wav_len = int(dur[0] * self.sr)
        T = (wav_len + self.samples_per_latent - 1) // self.samples_per_latent
        xt = rng.standard_normal((1, self.ldim * self.ccf, T)).astype(np.float32)
        lmask = H.get_latent_mask(np.array([wav_len]), self.base_chunk, self.ccf)
        xt = xt * lmask

        t0 = time.perf_counter()
        tot = np.array([steps], dtype=np.float32)
        for s in range(steps):
            xt, *_ = self.sess["vector_estimator"].run(None, {
                "noisy_latent": xt, "text_emb": emb, "style_ttl": self.style.ttl,
                "text_mask": mask, "latent_mask": lmask,
                "current_step": np.array([s], dtype=np.float32), "total_step": tot})
        t["vector_estimator"] = time.perf_counter() - t0
        return Plan(latent=xt, wav_len=wav_len, timings=t)

    # ---------- 後段：聲碼器 ----------
    def vocode(self, latent: np.ndarray) -> np.ndarray:
        wav, *_ = self.sess["vocoder"].run(None, {"latent": latent})
        return wav[0]

    def synth_full(self, text: str, **kw) -> tuple[np.ndarray, dict]:
        p = self.plan(text, **kw)
        t0 = time.perf_counter()
        wav = self.vocode(p.latent)[: p.wav_len]
        p.timings["vocoder"] = time.perf_counter() - t0
        return wav, p.timings
