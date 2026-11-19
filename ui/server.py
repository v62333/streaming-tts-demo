"""本機 demo 網頁伺服器（只用 Python 標準庫）。

聲音由本機喇叭播放（沿用 src/streamer.py 的環形緩衝與播放核心）；
網頁透過 Server-Sent Events 即時收到同一次播放的量測事件來畫圖，不放預錄動畫。

  .venv/bin/python ui/server.py            # 開 http://127.0.0.1:8765
  .venv/bin/python ui/server.py --silent   # 不出聲，用模擬音效卡節拍（測試用）
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from engine import Engine  # noqa: E402
from streamer import Control, stream  # noqa: E402

INT8 = {"vector_estimator": "models/int8/vector_estimator.int8.onnx"}
ENGINES: dict = {}


def load_mem_ref():
    """讀實驗 8 的實測峰值記憶體（每個模型在獨立行程量；伺服器同時載兩套，無法現場分開量）。"""
    import ast
    ref = {}
    try:
        for line in open(os.path.join(ROOT, "results", "08_memory.txt")):
            k, v = line.split(" ", 1)
            ref[{"int8ve": "int8"}.get(k, k)] = int(np.median([d["peak_rss_MB"] for d in ast.literal_eval(v)]))
    except (OSError, ValueError):
        pass
    return ref


MEM_REF = load_mem_ref()
ENGINE_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()        # 一次只跑一個，避免兩次量測互搶 CPU
RUNS: dict = {}                    # run_id -> {"q": Queue, "events": [], "audio": bytes|None}
CURRENT = {"ctl": None}            # 正在播放那一次的控制器（暫停／停止）
ARGS = None


def get_engine(variant: str, threads: int) -> Engine:
    key = (variant, threads)
    with ENGINE_LOCK:
        if key not in ENGINES:
            eng = Engine(threads=threads, model_overrides=INT8 if variant == "int8" else {})
            eng.synth_full("Warm up.")
            ENGINES[key] = eng
        return ENGINES[key]


def run_job(run_id: str, p: dict):
    r = RUNS[run_id]

    def emit(e):
        r["events"].append(e)
        r["q"].put(e)

    with RUN_LOCK:
        try:
            eng = get_engine(p["variant"], int(p.get("threads", 0)))
            ctl = Control()
            CURRENT["ctl"] = ctl
            epoch = time.time()
            emit({"type": "run", "run": run_id, "epoch": epoch, **p})
            st = stream(eng, p["text"], policy=p["mode"], steps=int(p["steps"]),
                        prefill_s=float(p.get("prefill", 0.2)),
                        player="sim" if ARGS.silent else "device",
                        on_event=emit, record=True, control=ctl)
            # 把實際送給喇叭的聲音串起來（影片配音用，跟畫面同一條時間軸）
            rec = st.core.recorded
            if rec:
                t0 = rec[0][0]
                pcm = np.concatenate([b for _, b in rec])
                buf = io.BytesIO()
                sf.write(buf, pcm, eng.sr, format="WAV")
                r["audio"] = buf.getvalue()
                r["audio_offset_s"] = t0
            summ = st.summary()
            gen, aud = sum(st.seg_gen_s), sum(st.seg_audio_s)
            summ["rtf"] = round(gen / aud, 3) if aud else None
            summ["mem_ref_mb"] = MEM_REF.get(p["variant"])
            emit({"type": "done", "summary": summ})
        except Exception as ex:  # 讓畫面看得到錯誤，不要卡住
            emit({"type": "error", "message": repr(ex)})
        finally:
            CURRENT["ctl"] = None
            r["q"].put(None)


def _csv(name):
    path = os.path.join(ROOT, "results", name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_experiments():
    """實驗 10～12 的結果（預先量測、讀結果檔，不是現場跑）。"""
    rt = _csv("10_runtime_options.csv")
    order = []
    for r in rt:
        if r["config"] not in order:
            order.append(r["config"])
    runtime = []
    for c in order:
        g = [r for r in rt if r["config"] == c]
        med = lambda k: statistics.median(float(r[k]) for r in g)  # noqa: E731
        runtime.append({"config": c, "n": len(g), **{k: med(k) for k in
                        ("load_s", "first_synth_s", "steady_synth_s", "after_load_MB", "peak_rss_MB")}})
    return {"runtime": runtime,
            "partitions": _csv("11_coreml_partitions.csv"),
            "coreml": _csv("11_coreml.csv"),
            "static": _csv("12_static_shape.csv"),
            "cache": _csv("12_coreml_cache.csv"),
            "cpu_ref": _csv("12_cpu_reference.csv")}


COLD_CHILD = r'''
import json, os, sys, time
sys.path.insert(0, "src")
t0 = time.perf_counter()
from engine import Engine
imp = time.perf_counter() - t0
t0 = time.perf_counter()
eng = Engine(model_overrides={"vector_estimator": "models/int8/vector_estimator.int8.onnx"})
load = time.perf_counter() - t0
t0 = time.perf_counter()
eng.synth_full("Turn left.", steps=8)
first = time.perf_counter() - t0
print(json.dumps({"import_s": round(imp, 3), "load_s": round(load, 3),
                  "per_model_s": {k: round(v, 3) for k, v in eng.load_s.items()},
                  "first_synth_s": round(first, 3)}))
'''


def cold_start():
    """開一個全新的行程：從零載入 INT8 版本並合成第一句（跟車子剛發動一樣）。"""
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", COLD_CHILD], capture_output=True, text=True, cwd=ROOT, timeout=120)
    total = time.perf_counter() - t0
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode or not line:
        return {"error": (r.stderr or "failed")[-300:]}
    return {**json.loads(line[-1]), "process_total_s": round(total, 3)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "ui", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/events":
            r = RUNS.get(q.get("run", [""])[0])
            if not r:
                return self._send(404, b"no such run", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while True:
                e = r["q"].get()
                if e is None:
                    break
                self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
                self.wfile.flush()
            return
        if u.path == "/api/audio":
            r = RUNS.get(q.get("run", [""])[0])
            if not r or not r.get("audio"):
                return self._send(404, b"no audio", "text/plain")
            return self._send(200, r["audio"], "audio/wav")
        if u.path == "/api/runlog":
            r = RUNS.get(q.get("run", [""])[0])
            if not r:
                return self._send(404, b"{}", "application/json")
            body = {"events": r["events"], "audio_offset_s": r.get("audio_offset_s")}
            return self._send(200, json.dumps(body).encode(), "application/json")
        if u.path == "/api/experiments":
            return self._send(200, json.dumps(load_experiments()).encode(), "application/json")
        if u.path == "/api/health":
            return self._send(200, b"ok", "text/plain")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        p = json.loads(self.rfile.read(n) or b"{}")
        if path == "/api/control":
            ctl = CURRENT["ctl"]
            if ctl is None:
                return self._send(409, b'{"error":"idle"}', "application/json")
            a = p.get("action")
            if a == "pause":
                ctl.pause.set()
            elif a == "resume":
                ctl.pause.clear()
            elif a == "stop":
                ctl.cancel.set()
                ctl.pause.clear()
            else:
                return self._send(400, b'{"error":"bad action"}', "application/json")
            return self._send(200, b'{"ok":true}', "application/json")
        if path == "/api/coldstart":
            if not RUN_LOCK.acquire(blocking=False):   # 正在播放時不量，避免互搶 CPU
                return self._send(409, b'{"error":"busy"}', "application/json")
            try:
                return self._send(200, json.dumps(cold_start()).encode(), "application/json")
            finally:
                RUN_LOCK.release()
        if path != "/api/run":
            return self._send(404, b"not found", "text/plain")
        p["text"] = (p.get("text") or "").strip()[:600]
        if not p["text"]:
            return self._send(400, b'{"error":"empty text"}', "application/json")
        if p.get("mode") not in ("whole", "sentence", "first_clause"):
            p["mode"] = "sentence"
        if p.get("variant") not in ("fp32", "int8"):
            p["variant"] = "int8"
        p["steps"] = min(max(int(p.get("steps", 8)), 1), 16)
        if RUN_LOCK.locked():
            return self._send(409, b'{"error":"busy"}', "application/json")
        run_id = uuid.uuid4().hex[:8]
        RUNS[run_id] = {"q": queue.Queue(), "events": [], "audio": None}
        threading.Thread(target=run_job, args=(run_id, p), daemon=True).start()
        return self._send(200, json.dumps({"run": run_id}).encode(), "application/json")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--silent", action="store_true")
    ARGS = ap.parse_args()
    print("loading models (fp32 + int8)...", flush=True)
    get_engine("fp32", 0)
    get_engine("int8", 0)
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    print(f"ready: http://127.0.0.1:{ARGS.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
