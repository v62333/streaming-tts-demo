"""固定大小環形緩衝（單一生產者、單一消費者）。

設計取捨：
- 寫入端（推論執行緒）滿了就「等」：車載語音不能丟音，所以用背壓（backpressure），不丟資料。
- 讀取端（播放回呼）永遠「不等」：音效卡回呼若被卡住會整段爆音，
  所以資料不夠時立刻回傳靜音並記一次斷音（underrun）。
- Python 版用一把鎖驗證時序與邏輯；產品版應改 C/C++ 無鎖 SPSC（只用原子讀寫指標）。
"""
from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int):
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.cap = capacity
        self.r = 0          # 累計已讀樣本數（單調遞增，取模得位置）
        self.w = 0          # 累計已寫樣本數
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.closed = False  # 生產者宣告不再寫入

    def level(self) -> int:
        return self.w - self.r

    def write(self, x: np.ndarray, timeout: float | None = None) -> None:
        """阻塞寫入整段；空間不夠就分批寫、等消費者騰出空間。"""
        x = np.asarray(x, dtype=np.float32)
        i = 0
        while i < len(x):
            with self.not_full:
                while self.cap - (self.w - self.r) == 0:
                    if not self.not_full.wait(timeout):
                        raise TimeoutError("ring buffer full too long")
                free = self.cap - (self.w - self.r)
                n = min(free, len(x) - i)
                pos = self.w % self.cap
                first = min(n, self.cap - pos)
                self.buf[pos:pos + first] = x[i:i + first]
                self.buf[0:n - first] = x[i + first:i + n]
                self.w += n
                i += n

    def read_nowait(self, n: int) -> tuple[np.ndarray, int]:
        """非阻塞讀 n 個樣本；回傳 (資料, 實際讀到幾個)。不足處補 0。"""
        out = np.zeros(n, dtype=np.float32)
        with self.lock:
            avail = self.w - self.r
            m = min(avail, n)
            pos = self.r % self.cap
            first = min(m, self.cap - pos)
            out[:first] = self.buf[pos:pos + first]
            out[first:m] = self.buf[0:m - first]
            self.r += m
            if m:
                self.not_full.notify()
        return out, m

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.not_full.notify_all()
