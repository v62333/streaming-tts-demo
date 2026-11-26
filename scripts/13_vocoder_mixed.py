"""實驗 13：聲碼器混合精度（採用版本）。

實驗 6 的消融結果：33 層卷積裡，只要第 1 層（輸入層 embed）和最後 2 層（輸出頭 head）留 FP32，
錯字率就回到 0%。這支腳本產生正式採用的版本：這 3 層 FP32、其餘 30 層 INT8。
需要先跑 scripts/04_quantize.py（產生 vocoder.pre.onnx）。
發版關卡：python scripts/release_gate.py int8_mixed
"""
import os

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

PRE = "models/int8/vocoder.pre.onnx"
OUT = "models/int8/vocoder.int8_mixed3.onnx"

convs = [n.name for n in onnx.load(PRE).graph.node if n.op_type == "Conv"]
assert len(convs) == 33, f"預期 33 層卷積，實際 {len(convs)}"
keep = convs[:1] + convs[-2:]          # 輸入層 1 層＋輸出頭 2 層
quantize_dynamic(PRE, OUT, weight_type=QuantType.QInt8,
                 op_types_to_quantize=["Conv"], nodes_to_exclude=keep)
print("FP32:", keep)
print("INT8:", len(convs) - len(keep), "層 →", OUT, os.path.getsize(OUT) // 2**20, "MB")
