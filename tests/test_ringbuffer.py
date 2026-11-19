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

def test_close_wakes_blocked_writer():
    from ringbuffer import BufferClosed
    rb = RingBuffer(4); rb.write(np.ones(4)); err = []
    def w():
        try: rb.write(np.ones(2))
        except BufferClosed: err.append("closed")
    t = threading.Thread(target=w); t.start(); time.sleep(0.05)
    rb.close(); t.join(1.0)
    assert not t.is_alive() and err == ["closed"]


def test_pause_is_not_underrun_and_stop_ends():
    from streamer import Control, PlaybackCore, PlaybackStats
    rb = RingBuffer(100); st = PlaybackStats(t_submit=time.perf_counter())
    done = threading.Event(); ctl = Control()
    core = PlaybackCore(rb, sr=10, block=5, prefill=0, stats=st, producer_done=done, control=ctl)
    rb.write(np.ones(5)); core.tick()            # 開播
    ctl.pause.set()
    for _ in range(4): core.tick()               # 暫停中：緩衝空了也不算斷音
    assert st.underruns == 0 and st.paused_s == 2.0
    ctl.cancel.set(); core.tick()
    assert st.stopped and core.finished.is_set()
