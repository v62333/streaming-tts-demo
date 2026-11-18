"""實驗 4：INT8 量化（ONNX Runtime 動態量化，權重 INT8、激活值執行時量化）。

只量化佔 95% 時間的 vector_estimator；聲碼器另外做一份，用來看「量化聲碼器」對音質的傷害。
動態量化不需要校正集；若要做靜態量化（需校正集），之後另開實驗。
"""
import os

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

SRC = "models/supertonic-3/onnx"
DST = "models/int8"
os.makedirs(DST, exist_ok=True)

for name in ("vector_estimator", "vocoder"):
    src = f"{SRC}/{name}.onnx"
    pre = f"{DST}/{name}.pre.onnx"
    out = f"{DST}/{name}.int8.onnx"
    try:
        quant_pre_process(src, pre, skip_symbolic_shape=True)
    except Exception as e:  # 前處理失敗就直接用原檔，記下原因
        print(f"[{name}] pre-process skipped: {e!r}")
        pre = src
    quantize_dynamic(pre, out, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm", "Conv"])
    print(name, os.path.getsize(src) // 2**20, "MB ->", os.path.getsize(out) // 2**20, "MB")
