"""雲端串流 TTS 服務（容器內跑；只用標準庫＋onnxruntime/numpy）。

POST /v1/tts  {"text": "...", "steps": 8}
  → 200，Transfer-Encoding: chunked，Content-Type: audio/L16;rate=44100;channels=1
    每生成完一句就送出一塊 16-bit PCM（逐句串流，跟本機 demo 同一套切段）
GET  /healthz   → 模型載入完成才回 200（給容器編排的存活／就緒檢查）
GET  /metrics   → 純文字計數器（請求數、進行中、排隊中、被拒、首段延遲分位數）

環境變數：
  MODEL_VARIANT   int8 | fp32（預設 int8）
  ORT_THREADS     每個推論用幾條執行緒（預設 1）
  MAX_CONCURRENT  同時最多幾個請求在生成（預設 2）；其餘排隊
  MAX_QUEUE       排隊上限（預設 8）；再多直接回 503（保護已在服務中的請求不被拖垮）
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from engine import Engine  # noqa: E402
from streamer import split_text  # noqa: E402

VARIANT = os.environ.get("MODEL_VARIANT", "int8")
THREADS = int(os.environ.get("ORT_THREADS", "1"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "8"))

ENGINE: Engine | None = None
SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT)
LOCK = threading.Lock()
STATS = {"requests": 0, "in_flight": 0, "queued": 0, "rejected": 0, "errors": 0, "ttfa_ms": []}


def load():
    global ENGINE
    ov = {"vector_estimator": "models/int8/vector_estimator.int8.onnx"} if VARIANT == "int8" else {}
    eng = Engine(threads=THREADS, model_overrides=ov)
    eng.synth_full("Warm up.")
    ENGINE = eng


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if xs else None


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/healthz":
            return self._json(200 if ENGINE else 503, {"ready": ENGINE is not None, "variant": VARIANT})
        if self.path == "/metrics":
            with LOCK:
                s = dict(STATS)
                t = list(STATS["ttfa_ms"])
            lines = [f"tts_requests_total {s['requests']}", f"tts_in_flight {s['in_flight']}",
                     f"tts_queued {s['queued']}", f"tts_rejected_total {s['rejected']}",
                     f"tts_errors_total {s['errors']}",
                     f"tts_ttfa_ms_p50 {pct(t, 50)}", f"tts_ttfa_ms_p95 {pct(t, 95)}",
                     f"tts_config{{variant=\"{VARIANT}\",threads=\"{THREADS}\",max_concurrent=\"{MAX_CONCURRENT}\"}} 1"]
            b = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/tts":
            return self._json(404, {"error": "not found"})
        t0 = time.perf_counter()
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = (body.get("text") or "").strip()[:1000]
        steps = min(max(int(body.get("steps", 8)), 1), 16)
        if not text or ENGINE is None:
            return self._json(400 if text == "" else 503, {"error": "empty text" if not text else "not ready"})
        with LOCK:
            STATS["requests"] += 1
            if STATS["queued"] >= MAX_QUEUE:
                STATS["rejected"] += 1
                reject = True
            else:
                STATS["queued"] += 1
                reject = False
        if reject:   # 卸載：排隊已滿，與其讓所有人一起變慢，不如明確拒絕這一個
            return self._json(503, {"error": "overloaded"})
        SLOTS.acquire()
        with LOCK:
            STATS["queued"] -= 1
            STATS["in_flight"] += 1
        queue_ms = (time.perf_counter() - t0) * 1000
        try:
            segs = split_text(text, "sentence")
            self.send_response(200)
            self.send_header("Content-Type", f"audio/L16;rate={ENGINE.sr};channels=1")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Queue-Ms", f"{queue_ms:.1f}")
            self.send_header("X-Segments", str(len(segs)))
            self.end_headers()
            for i, seg in enumerate(segs):
                wav, _ = ENGINE.synth_full(seg, steps=steps)
                pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
                self.wfile.write(f"{len(pcm):X}\r\n".encode() + pcm + b"\r\n")
                self.wfile.flush()
                if i == 0:
                    with LOCK:
                        STATS["ttfa_ms"].append((time.perf_counter() - t0) * 1000)
                        STATS["ttfa_ms"] = STATS["ttfa_ms"][-2000:]
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            with LOCK:
                STATS["errors"] += 1
        finally:
            SLOTS.release()
            with LOCK:
                STATS["in_flight"] -= 1


def main():
    port = int(os.environ.get("PORT", "8800"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"listening :{port} (loading {VARIANT}, threads={THREADS}, max_concurrent={MAX_CONCURRENT})", flush=True)
    load()
    print("ready", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
