"""串流合成：推論執行緒逐段生成 → 環形緩衝 → 播放端固定節拍消費。

播放端兩種：
- SimulatedPlayer：不出聲，用牆鐘每 block 取一次（跟音效卡回呼同節奏），給量測用。
- 真實播放：sounddevice 的 OutputStream 回呼（demo 用）。
兩者共用同一套「預填量、斷音判定、水位記錄」邏輯（PlaybackCore）。
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ringbuffer import RingBuffer


# ---------------- 切段策略 ----------------
def split_text(text: str, policy: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if policy == "whole":
        return [text]
    if policy == "sentence":
        parts = re.split(r"(?<=[.!?])\s+", text)
    elif policy == "clause":
        parts = re.split(r"(?<=[.!?,;:])\s+", text)
    elif policy == "first_clause":
        # 只把第一句再切到第一個逗號（搶首段延遲），之後按整句
        sents = re.split(r"(?<=[.!?])\s+", text)
        head = re.split(r"(?<=[,;:])\s+", sents[0], maxsplit=1)
        parts = head + sents[1:]
    else:
        raise ValueError(policy)
    return [p for p in parts if p.strip()]


# ---------------- 播放核心 ----------------
@dataclass
class PlaybackStats:
    t_submit: float = 0.0
    t_first_audio: float | None = None      # 播放端第一次送出非靜音的時刻
    t_end: float | None = None
    underruns: int = 0                      # 播放開始後，資料不足的 block 次數
    underrun_s: float = 0.0                 # 斷音總長（秒）
    level_trace: list = field(default_factory=list)   # (t, 水位秒數)
    seg_ready: list = field(default_factory=list)     # 每段生成完成時刻
    seg_gen_s: list = field(default_factory=list)     # 每段生成耗時
    seg_audio_s: list = field(default_factory=list)   # 每段聲音長度

    def summary(self) -> dict:
        g = np.array(self.seg_gen_s) if self.seg_gen_s else np.zeros(1)
        return {
            "first_audio_ms": round((self.t_first_audio - self.t_submit) * 1000, 1)
            if self.t_first_audio else None,
            "underruns": self.underruns,
            "underrun_ms": round(self.underrun_s * 1000, 1),
            "segments": len(self.seg_gen_s),
            "seg_gen_p50_ms": round(float(np.percentile(g, 50)) * 1000, 1),
            "seg_gen_max_ms": round(float(g.max()) * 1000, 1),
            "audio_s": round(sum(self.seg_audio_s), 2),
            "wall_s": round((self.t_end - self.t_submit), 2) if self.t_end else None,
        }


class PlaybackCore:
    """播放端邏輯：未開播時等預填量；開播後每 block 取資料，不足即算斷音。"""

    def __init__(self, rb: RingBuffer, sr: int, block: int, prefill: int,
                 stats: PlaybackStats, producer_done: threading.Event,
                 on_event=None, record: bool = False):
        self.rb, self.sr, self.block, self.prefill = rb, sr, block, prefill
        self.st, self.done = stats, producer_done
        self.started = False
        self.finished = threading.Event()
        self.emit = on_event or (lambda e: None)
        self.recorded = [] if record else None   # (相對送出時刻, 實際送給喇叭的 block)
        self._last_level_emit = -1.0

    def tick(self) -> np.ndarray:
        now = time.perf_counter()
        t = now - self.st.t_submit
        lvl = self.rb.level()
        self.st.level_trace.append((t, lvl / self.sr))
        if t - self._last_level_emit >= 0.05:
            self._last_level_emit = t
            self.emit({"type": "level", "t": round(t, 3), "level_s": round(lvl / self.sr, 3)})
        if not self.started:
            if lvl >= self.prefill or (self.done.is_set() and lvl > 0):
                self.started = True
                self.st.t_first_audio = now
                self.emit({"type": "first_audio", "t": round(t, 3)})
            else:
                return self._out(t, np.zeros(self.block, dtype=np.float32))
        data, got = self.rb.read_nowait(self.block)
        if got < self.block:
            if self.done.is_set() and self.rb.level() == 0:
                if not self.finished.is_set():
                    self.st.t_end = now + got / self.sr
                    self.finished.set()
                    self.emit({"type": "end", "t": round(self.st.t_end - self.st.t_submit, 3)})
            else:
                self.st.underruns += 1
                self.st.underrun_s += (self.block - got) / self.sr
                self.emit({"type": "underrun", "t": round(t, 3),
                           "missing_s": round((self.block - got) / self.sr, 4)})
        return self._out(t, data)

    def _out(self, t, data):
        if self.recorded is not None:
            self.recorded.append((t, data.copy()))
        return data


class SimulatedPlayer(threading.Thread):
    """用牆鐘模擬音效卡：每 block/sr 秒取一次資料（不累積漂移）。"""

    def __init__(self, core: PlaybackCore):
        super().__init__(daemon=True)
        self.core = core

    def run(self):
        period = self.core.block / self.core.sr
        nxt = time.perf_counter()
        while not self.core.finished.is_set():
            self.core.tick()
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)


# ---------------- 生產者 ----------------
def produce(engine, segments: list[str], rb: RingBuffer, stats: PlaybackStats,
            done: threading.Event, steps: int, gap_s: float, write_block: int,
            emit=lambda e: None):
    gap = np.zeros(int(gap_s * engine.sr), dtype=np.float32)
    for i, seg in enumerate(segments):
        t0 = time.perf_counter()
        emit({"type": "seg_start", "i": i, "t": round(t0 - stats.t_submit, 3), "text": seg})
        wav, _ = engine.synth_full(seg, steps=steps)
        t1 = time.perf_counter()
        emit({"type": "seg_done", "i": i, "t": round(t1 - stats.t_submit, 3),
              "gen_s": round(t1 - t0, 3), "audio_s": round(len(wav) / engine.sr, 3)})
        stats.seg_gen_s.append(t1 - t0)
        stats.seg_ready.append(t1 - stats.t_submit)
        stats.seg_audio_s.append(len(wav) / engine.sr)
        if i:
            rb.write(gap)
        for j in range(0, len(wav), write_block):
            rb.write(wav[j:j + write_block])
    done.set()
    rb.close()


def stream(engine, text: str, policy: str = "sentence", steps: int = 8,
           prefill_s: float = 0.2, capacity_s: float = 5.0, block: int = 1024,
           gap_s: float = 0.15, player: str = "sim", on_event=None,
           record: bool = False) -> PlaybackStats:
    sr = engine.sr
    rb = RingBuffer(int(capacity_s * sr))
    st = PlaybackStats(t_submit=time.perf_counter())
    done = threading.Event()
    core = PlaybackCore(rb, sr, block, int(prefill_s * sr), st, done, on_event, record)
    st.core = core
    segs = split_text(text, policy)
    emit = on_event or (lambda e: None)
    emit({"type": "plan", "segments": segs, "sr": sr, "capacity_s": capacity_s})

    if player == "sim":
        p = SimulatedPlayer(core)
        p.start()
        produce(engine, segs, rb, st, done, steps, gap_s, write_block=block * 4, emit=emit)
        p.join()
    else:
        import sounddevice as sd

        def cb(outdata, frames, t, status):
            outdata[:, 0] = core.tick()
        with sd.OutputStream(samplerate=sr, channels=1, blocksize=block,
                             dtype="float32", callback=cb):
            produce(engine, segs, rb, st, done, steps, gap_s, write_block=block * 4, emit=emit)
            core.finished.wait()
            time.sleep(0.3)
    return st
