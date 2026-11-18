"""即時 demo：真的從喇叭播出來，終端機每 0.1 秒畫一次緩衝水位。

用法：
  .venv/bin/python scripts/demo_live.py                      # 預設 INT8、逐句串流
  .venv/bin/python scripts/demo_live.py --variant fp32 --policy whole   # 對照：整句做完才播
  .venv/bin/python scripts/demo_live.py --text "Your own sentence. Another one."
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from engine import Engine  # noqa: E402
import streamer  # noqa: E402

DEFAULT = ("Traffic is heavy on the route ahead, so I found a faster way that saves about "
           "twelve minutes. It goes through the city center, and there is one toll road. "
           "Would you like me to switch to the new route, or keep the current one?")

ap = argparse.ArgumentParser()
ap.add_argument("--text", default=DEFAULT)
ap.add_argument("--variant", default="int8ve", choices=["fp32", "int8ve"])
ap.add_argument("--policy", default="sentence")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--prefill", type=float, default=0.2)
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()

ov = {"vector_estimator": "models/int8/vector_estimator.int8.onnx"} if a.variant == "int8ve" else {}
eng = Engine(threads=a.threads, model_overrides=ov)
eng.synth_full("Warm up.")

result = {}
stop = threading.Event()


def run():
    result["st"] = streamer.stream(eng, a.text, policy=a.policy, steps=a.steps,
                                   prefill_s=a.prefill, player="device")
    stop.set()


# 為了畫水位，偷看 stream() 內部的 PlaybackStats：用 level_trace 的最後一筆
_orig = streamer.PlaybackStats
live = {}


class Tapped(_orig):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        live["st"] = self


streamer.PlaybackStats = Tapped
th = threading.Thread(target=run, daemon=True)
th.start()
cap_s = 5.0
while not stop.is_set():
    st = live.get("st")
    if st and st.level_trace:
        t, lvl = st.level_trace[-1]
        bar = "#" * int(40 * lvl / cap_s)
        state = "PLAYING" if st.t_first_audio else "waiting"
        sys.stdout.write(f"\r t={t:5.2f}s  buffer {lvl:4.2f}s |{bar:<40}| {state}  underruns={st.underruns:3d}")
        sys.stdout.flush()
    time.sleep(0.1)
print()
print(result["st"].summary())
