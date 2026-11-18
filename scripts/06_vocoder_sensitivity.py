"""實驗 6：聲碼器 INT8 壞掉，是哪幾層造成的？逐一保留某些層為 FP32 來定位。"""
import csv, importlib.util, os, sys
import numpy as np, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("q", "scripts/05_quality.py"); q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
from engine import Engine
import librosa
from faster_whisper import WhisperModel

PRE = "models/int8/vocoder.pre.onnx"
convs = [n.name for n in onnx.load(PRE).graph.node if n.op_type == "Conv"]
CASES = {"keep_none": [], "keep_first1": convs[:1], "keep_last1": convs[-1:],
         "keep_last2_head": convs[-2:], "keep_first1_last2": convs[:1] + convs[-2:]}
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
ve = "models/int8/vector_estimator.int8.onnx"
ref_eng = Engine(model_overrides={"vector_estimator": ve})
refs = {i: q.logmel(ref_eng.synth_full(s, steps=8, seed=i)[0], ref_eng.sr) for i, s in enumerate(q.SENTS)}
rows = []
for name, keep in CASES.items():
    out = f"models/int8/vocoder.sens_{name}.onnx"
    quantize_dynamic(PRE, out, weight_type=QuantType.QInt8, op_types_to_quantize=["Conv"], nodes_to_exclude=keep)
    eng = Engine(model_overrides={"vector_estimator": ve, "vocoder": out})
    errs = words = 0; d = []
    for i, s in enumerate(q.SENTS):
        wav, _ = eng.synth_full(s, steps=8, seed=i)
        hyp = " ".join(x.text for x in asr.transcribe(librosa.resample(wav, orig_sr=eng.sr, target_sr=16000), language="en", beam_size=5)[0])
        e, n = q.wer(s, hyp); errs += e; words += n
        m = q.logmel(wav, eng.sr); L = min(m.shape[1], refs[i].shape[1]); d.append(float(np.abs(m[:, :L] - refs[i][:, :L]).mean()))
    r = {"case": name, "fp32_layers": len(keep), "WER_%": round(100 * errs / words, 1), "mel_dist_vs_fp32voc": round(float(np.mean(d)), 3), "size_MB": os.path.getsize(out) // 2**20}
    rows.append(r); print(r, flush=True); os.remove(out)
with open("results/06_vocoder_sensitivity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
