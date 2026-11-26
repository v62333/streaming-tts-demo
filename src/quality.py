"""客觀品質指標（給實驗腳本與發版標準共用）。

- WER：whisper 把合成語音轉回文字的錯字率（數字已正規化：200＝two hundred）
- logmel 距離：與參考輸出的 log-mel 頻譜平均差
"""
import re

import numpy as np

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


_ASR = None


def transcribe(wav, sr):
    """whisper small.en（CPU、INT8）；第一次呼叫才載入。"""
    global _ASR
    import librosa
    from faster_whisper import WhisperModel
    if _ASR is None:
        _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    segs, _ = _ASR.transcribe(w16, language="en", beam_size=5)
    return " ".join(x.text for x in segs).strip()


def mel_dist(a, b, sr):
    ma, mb = logmel(a, sr), logmel(b, sr)
    n = min(ma.shape[1], mb.shape[1])
    return float(np.mean(np.abs(ma[:, :n] - mb[:, :n])))
