"""實驗 8：峰值記憶體（每個版本開獨立行程量，避免互相污染）。"""
import json, os, subprocess, sys
CODE = r'''
import os, sys, resource, psutil
sys.path.insert(0, "src")
from engine import Engine
ov = json.loads(sys.argv[1])
p = psutil.Process()
base = p.memory_info().rss
eng = Engine(model_overrides=ov)
loaded = p.memory_info().rss
eng.synth_full("Traffic is heavy on the route ahead, so I found a faster way that saves about twelve minutes. It goes through the city center, and there is one toll road.", steps=8)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS: bytes
print(json.dumps({"after_load_MB": round((loaded-base)/2**20), "peak_rss_MB": round(peak/2**20)}))
'''.replace("import os, sys", "import os, sys, json")
V = {"fp32": {}, "int8ve": {"vector_estimator": "models/int8/vector_estimator.int8.onnx"}}
for k, ov in V.items():
    r = [json.loads(subprocess.run([sys.executable, "-c", CODE, json.dumps(ov)], capture_output=True, text=True).stdout.strip().splitlines()[-1]) for _ in range(3)]
    print(k, r)
