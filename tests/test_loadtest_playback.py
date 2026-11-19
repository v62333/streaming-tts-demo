import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from loadtest import simulate_playback

def test_no_stall_when_chunks_arrive_in_time():
    # 第一塊 0.5s 到、長 2s；第二塊 1.5s 到（還在播第一塊）
    assert simulate_playback([(0.5, 2.0), (1.5, 1.0)])[:3] == (0.5, 0, 0.0)

def test_stall_when_next_chunk_late():
    first, n, s, audio = simulate_playback([(0.5, 1.0), (2.0, 1.0)])   # 1.5s 播完，2.0s 才到
    assert (first, n, round(s, 3), audio) == (0.5, 1, 0.5, 2.0)
