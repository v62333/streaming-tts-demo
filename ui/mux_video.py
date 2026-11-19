"""把錄好的畫面（webm，無聲）和每次播放實際送給喇叭的聲音合成 mp4，並自動對齊。

對齊方法：Playwright 的影片起點只能近似，所以用畫面本身校正——
找出畫面上「第一次出現綠色播放條」的時刻，對齊第一次播放的「開口」聲音時刻，
算出位移後套用到所有片段（同一條影片時鐘）。

  .venv/bin/python ui/mux_video.py results/video/runs.json
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

GREEN = np.array([27, 175, 122])  # 播放條 #1baf7a


def first_green(webm: str, start: float, dur: float = 15.0, fps: int = 50) -> float:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", webm,
                        "-vf", f"fps={fps}", f"{d}/f_%05d.png"], check=True)
        for i, f in enumerate(sorted(os.listdir(d))):
            a = np.asarray(Image.open(os.path.join(d, f)).convert("RGB")).astype(int)
            if (np.abs(a - GREEN) < 25).all(axis=-1).sum() > 200:
                return start + i / fps
    raise RuntimeError("找不到播放條，無法校正")


def main(meta_path: str):
    m = json.load(open(meta_path))
    out_dir = os.path.dirname(meta_path)
    runs = m["runs"]   # 每筆：wav, epoch, audio_offset_s, first_audio_t
    t0 = m["t_video0"]
    # 未校正時，每段聲音在影片中的起點（秒）
    raw = [r["epoch"] + r["audio_offset_s"] - t0 for r in runs]
    r0 = runs[0]
    placed_first = raw[0] + (r0["first_audio_t"] - r0["audio_offset_s"])
    seen = first_green(m["webm"], max(0.0, placed_first - 6))
    shift = seen - placed_first
    print(f"開口：聲音原本放在 {placed_first:.3f}s，畫面出現在 {seen:.3f}s → 位移 {shift:+.3f}s")

    inputs, filt = ["-i", m["webm"]], []
    for k, (r, st) in enumerate(zip(runs, raw), 1):
        inputs += ["-i", r["wav"]]
        filt.append(f"[{k}:a]adelay={max(0, round((st + shift) * 1000))}:all=1[a{k}]")
    n = len(runs)
    mix = ";".join(filt) + ";" + "".join(f"[a{k}]" for k in range(1, n + 1)) + \
        f"amix=inputs={n}:normalize=0[aout]"
    mp4 = os.path.join(out_dir, "streaming_tts_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", mix,
                    "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-c:a", "aac", "-b:a", "128k", mp4], check=True)
    print("wrote", mp4)


if __name__ == "__main__":
    main(sys.argv[1])
