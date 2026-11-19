"""壓測雲端串流服務：同時 N 個使用者，每人量「多久聽到第一個字」和「會不會斷音」。

每個客戶端各自模擬喇叭播放（跟本機 demo 同一套規則）：
  收到第一塊就開播；之後若下一塊抵達時，前面的聲音已經播完 → 算一次斷音，停頓長度＝空等的時間。
因為服務是逐句送，斷音＝句子之間的停頓被拉長。

  python scripts/loadtest.py --url http://127.0.0.1:8800 --levels 1 2 4 8 --tag cpus4_mc2
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import threading
import time
from urllib.parse import urlparse

import numpy as np

TEXTS = [
    "In two hundred meters, turn right onto Main Street. Then keep left at the fork. Your destination will be on the right.",
    "Your battery is at twenty percent. The nearest charging station is four kilometers ahead, on the right side of the highway.",
    "New message from Anna. She says she is running late and will meet you at the restaurant at seven thirty. Do you want to reply?",
    "Traffic is heavy on the route ahead, so I found a faster way. It goes through the city center, and there is one toll road.",
    "The temperature outside is minus three degrees. The road may be icy, so please drive carefully. Seat heating is now on.",
    "Calling Mom on her mobile phone. The call will start in a moment. Say cancel if you want to stop the call.",
    "Your next meeting starts in fifteen minutes. It is with the design team. Do you want me to send a message that you are on the way?",
    "Low tire pressure detected on the rear left tire. Please check the tire at the next stop. The nearest service station is two kilometers away.",
]


def one_request(url, text, steps, out):
    u = urlparse(url)
    t0 = time.perf_counter()
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
        c.request("POST", "/v1/tts", json.dumps({"text": text, "steps": steps}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            out.update(status=r.status, t0=t0)
            r.read()
            return
        sr = int(r.getheader("Content-Type").split("rate=")[1].split(";")[0])
        queue_ms = float(r.getheader("X-Queue-Ms", "0"))
        chunks = []  # (抵達時間, 聲音秒數)；http.client 會自動解 chunked，這裡用 read1 盡快拿到
        buf = b""
        nseg = int(r.getheader("X-Segments", "1"))
        while True:
            b = r.read1(1 << 16) if hasattr(r, "read1") else r.read(1 << 16)
            if not b:
                break
            buf += b
            chunks.append((time.perf_counter() - t0, len(b) / 2 / sr))
        out.update(status=200, t0=t0, chunks=chunks, sr=sr, queue_ms=queue_ms, nseg=nseg)
    except Exception as e:  # noqa: BLE001
        out.update(status=-1, error=repr(e), t0=t0)


def simulate_playback(chunks):
    """回傳 (首段延遲, 斷音次數, 斷音總長, 聲音總長)。小於 10 ms 的空隙不算。"""
    if not chunks:
        return None, 0, 0.0, 0.0
    first = chunks[0][0]
    clock = first
    stalls, stall_s, audio = 0, 0.0, 0.0
    for arr, dur in chunks:
        if arr > clock + 0.010:      # 前面的聲音已播完、這塊還沒到 → 斷音
            stalls += 1
            stall_s += arr - clock
        clock = max(clock, arr) + dur
        audio += dur
    return first, stalls, stall_s, audio


def run_level(url, n, steps, rep):
    outs = [dict() for _ in range(n)]
    ths = [threading.Thread(target=one_request, args=(url, TEXTS[(i + rep) % len(TEXTS)], steps, outs[i]))
           for i in range(n)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ok = [o for o in outs if o.get("status") == 200]
    res = [simulate_playback(o["chunks"]) for o in ok]
    ttfa = [r[0] for r in res if r[0] is not None]
    audio_total = sum(r[3] for r in res)
    return {
        "users": n, "rep": rep, "ok": len(ok), "rejected_503": sum(o.get("status") == 503 for o in outs),
        "errors": sum(o.get("status") == -1 for o in outs),
        "ttfa_p50_s": round(float(np.median(ttfa)), 3) if ttfa else None,
        "ttfa_max_s": round(max(ttfa), 3) if ttfa else None,
        "users_with_stall": sum(r[1] > 0 for r in res),
        "stall_total_s": round(sum(r[2] for r in res), 2),
        "stall_max_per_user_s": round(max((r[2] for r in res), default=0), 2),
        "queue_ms_max": round(max((o.get("queue_ms", 0) for o in ok), default=0), 1),
        "audio_s_per_wall_s": round(audio_total / wall, 2),
        "wall_s": round(wall, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="results/09_loadtest.csv")
    a = ap.parse_args()
    run_level(a.url, 1, a.steps, 99)   # 暖機，不計
    rows = []
    for n in a.levels:
        for rep in range(a.repeat):
            r = {"tag": a.tag, **run_level(a.url, n, a.steps, rep)}
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
