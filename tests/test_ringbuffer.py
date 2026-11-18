import os, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ringbuffer import RingBuffer

def test_wraparound_integrity():
    rb = RingBuffer(7); src = np.arange(1000, dtype=np.float32); got = []
    def prod():
        for i in range(0, 1000, 5): rb.write(src[i:i+5])   # 會一直被背壓卡住
    t = threading.Thread(target=prod); t.start()
    while len(got) < 1000:
        d, m = rb.read_nowait(3); got.extend(d[:m]);
        if m == 0: time.sleep(0.0005)
    t.join(); assert np.array_equal(np.array(got), src)

def test_backpressure_blocks_writer():
    rb = RingBuffer(4); rb.write(np.ones(4))
    try: rb.write(np.ones(1), timeout=0.05); assert False, "should block"
    except TimeoutError: pass
    assert rb.level() == 4

def test_underread_pads_zero():
    rb = RingBuffer(8); rb.write(np.array([1, 2], dtype=np.float32))
    d, m = rb.read_nowait(4); assert m == 2 and list(d) == [1, 2, 0, 0]
