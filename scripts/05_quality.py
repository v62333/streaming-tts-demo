"""實驗 5：各版本（步數、INT8）的速度與品質。

品質兩個客觀指標（不靠自己耳朵打分）：
- WER：用 whisper（faster-whisper small.en）把合成語音轉回文字，算字詞錯誤率 → 念錯／念糊
- mel_dist：跟「FP32、8 步、同一個隨機種子」的參考輸出比對 log-mel 頻譜平均差 → 音色偏離多少
  （同一句、同種子，量化前後輸入雜訊相同，所以可直接比較；步數不同時只當參考）
"""
import csv
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from engine import Engine  # noqa: E402

SENTS = [
    "Turn left.",
    "In two hundred meters, turn right onto Main Street.",
    "Your battery is at twenty percent.",
    "The nearest charging station is four kilometers ahead, on the right side of the highway.",
    "Traffic is heavy on the route ahead, so I found a faster way that saves about twelve minutes.",
    "Would you like me to switch to the new route, or keep the current one?",
    "New message from Anna. She says she is running late.",
    "The temperature outside is minus three degrees, and the road may be icy.",
    "Calling Mom on her mobile phone.",
    "Please take over the steering wheel.",
]

VARIANTS = [  # 名稱, 步數, 覆寫的模型（seed 偏移：見 SEED_OFFSET）
    ("fp32_s8", 8, {}),
    ("fp32_s8_otherseed", 8, {}),   # 雜訊基準：同設定只換隨機種子，差多少算「正常變化」
    ("fp32_s4", 4, {}),
    ("fp32_s2", 2, {}),
    ("int8ve_s8", 8, {"vector_estimator": "models/int8/vector_estimator.int8.onnx"}),
    ("int8ve_s4", 4, {"vector_estimator": "models/int8/vector_estimator.int8.onnx"}),
    ("int8ve_voc_perchannel_s8", 8, {"vector_estimator": "models/int8/vector_estimator.int8.onnx",
                                     "vocoder": "models/int8/vocoder.int8_perchannel.onnx"}),
    ("int8ve_voc_keepends_s8", 8, {"vector_estimator": "models/int8/vector_estimator.int8.onnx",
                                   "vocoder": "models/int8/vocoder.int8_keepends.onnx"}),
    ("int8all_s8", 8, {"vector_estimator": "models/int8/vector_estimator.int8.onnx",
                       "vocoder": "models/int8/vocoder.int8.onnx"}),
]


def norm(s):
    # 數字轉英文字（whisper 會把 two hundred 寫成 200，那不是念錯）
    from num2words import num2words
    s = s.replace("%", " percent").replace(" km ", " kilometers ")
    s = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", s)
    s = re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " "))
    return s.split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1], len(r)


def logmel(wav, sr):
    import librosa
    m = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    return np.log(m + 1e-5)


def main():
    from faster_whisper import WhisperModel
    asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    os.makedirs("results/05_wavs", exist_ok=True)
    import soundfile as sf
    import librosa

    ref_mels = {}
    rows = []
    for name, steps, ov in VARIANTS:
        if any(not os.path.exists(p) for p in ov.values()):
            print("skip", name, "(model missing)")
            continue
        eng = Engine(model_overrides=ov)
        eng.synth_full("Warm up.", steps=steps)
        errs = words = 0
        dists, gen, aud = [], [], []
        for i, s in enumerate(SENTS):
            t0 = time.perf_counter()
            seed = i + (1000 if name.endswith("otherseed") else 0)
            wav, _ = eng.synth_full(s, steps=steps, seed=seed)
            gen.append(time.perf_counter() - t0)
            aud.append(len(wav) / eng.sr)
            sf.write(f"results/05_wavs/{name}_{i}.wav", wav, eng.sr)
            w16 = librosa.resample(wav, orig_sr=eng.sr, target_sr=16000)
            segs, _ = asr.transcribe(w16, language="en", beam_size=5)
            hyp = " ".join(x.text for x in segs)
            e, n = wer(s, hyp)
            errs += e
            words += n
            if e:
                print(f"  [{name}#{i}] ref={s!r} hyp={hyp.strip()!r}")
            m = logmel(wav, eng.sr)
            if name == "fp32_s8":
                ref_mels[i] = m
            r = ref_mels[i]
            L = min(r.shape[1], m.shape[1])
            dists.append(float(np.mean(np.abs(r[:, :L] - m[:, :L]))))
        row = {"variant": name, "steps": steps, "WER_%": round(100 * errs / words, 2),
               "mel_dist": round(float(np.mean(dists)), 3),
               "gen_ms_total": round(sum(gen) * 1000), "rtf": round(sum(gen) / sum(aud), 3)}
        rows.append(row)
        print(row, flush=True)
    with open("results/05_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
