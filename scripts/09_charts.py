"""把 results/ 的實測 CSV 畫成比較圖（results/charts/*.png）。數字全部讀 CSV，不手填。"""
import csv
import os
import statistics as S
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "results/charts"
os.makedirs(OUT, exist_ok=True)

# 配色：參考調色盤的分類色（固定順序）；紅色只給「壞掉」這種狀態
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRIT = "#d03b3b"
INK, INK2, GRID, SURF = "#1f1f1e", "#5c5b55", "#e4e3de", "#fcfcfb"

for f in ("PingFang TC", "Heiti TC", "Arial Unicode MS", "Hiragino Sans"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def read(p):
    return list(csv.DictReader(open(p)))


def save(fig, name, note):
    fig.text(0.01, -0.06, note, fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.savefig(f"{OUT}/{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------- 圖 1：時間花在哪 ----------
rows = read("results/01_stage_profile.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
labels = [f"{r['chars']} 字（{float(r['audio_s']):.1f} 秒聲音）" for r in rows]
ve = [float(r["vector_estimator"]) for r in rows]
voc = [float(r["vocoder"]) for r in rows]
oth = [float(r["text_proc"]) + float(r["duration"]) + float(r["text_encoder"]) for r in rows]
y = range(len(rows))
ax.barh(y, ve, color=BLUE, label="逐步生成", height=0.55, edgecolor=SURF, linewidth=2)
ax.barh(y, voc, left=ve, color=ORANGE, label="轉聲波（聲碼器）", height=0.55, edgecolor=SURF, linewidth=2)
ax.barh(y, oth, left=[a + b for a, b in zip(ve, voc)], color=AQUA, label="其他（長度預測＋文字編碼）",
        height=0.55, edgecolor=SURF, linewidth=2)
for i in y:
    tot = ve[i] + voc[i] + oth[i]
    ax.text(tot + 60, i, f"逐步生成佔 {100 * ve[i] / tot:.0f}%　聲碼器 {100 * voc[i] / tot:.0f}%",
            va="center", fontsize=9.5, color=INK)
ax.set_yticks(list(y), labels)
ax.invert_yaxis()
ax.set_xlabel("整句生成耗時（毫秒，中位數）")
ax.set_xlim(0, max(a + b + c for a, b, c in zip(ve, voc, oth)) * 1.55)
ax.set_title("時間幾乎全花在「逐步生成」，聲碼器只佔約 3%")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.grid(axis="y", visible=False)
save(fig, "1_stage_breakdown", "Mac M4 原生、FP32、8 步、n=5 中位數｜資料：results/01_stage_profile.csv")

# ---------- 圖 2：串流 vs 整句的首段延遲 ----------
rows = read("results/03_streaming_bench.csv")
g = defaultdict(list)
for r in rows:
    g[(r["threads"], r["steps"], r["policy"])].append(float(r["first_audio_ms"]))
pols = [("whole", "整句做完才播"), ("sentence", "逐句串流"), ("first_clause", "逐句＋首句切到逗號")]
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.36
for k, (th, name, col) in enumerate([("auto", "多執行緒", BLUE), ("1", "單執行緒", ORANGE)]):
    med = [S.median(g[(th, "8", p)]) for p, _ in pols]
    mx = [max(g[(th, "8", p)]) for p, _ in pols]
    xs = [i + (k - 0.5) * w for i in range(len(pols))]
    ax.bar(xs, med, width=w - 0.04, color=col, label=f"{name}（柱＝中位數）")
    ax.scatter(xs, mx, marker="_", s=260, linewidths=2.5, color=INK, zorder=3,
               label="最差一次（12 次中）" if k == 0 else None)
    for x, m in zip(xs, med):
        ax.text(x, m / 2, f"{m / 1000:.2f}s", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
ax.set_xticks(range(len(pols)), [n for _, n in pols])
ax.set_ylabel("首段延遲（毫秒）：送出文字 → 聽到第一個聲音")
ax.set_title("逐句串流讓開口時間縮短到約 1/3")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="x", visible=False)
save(fig, "2_first_audio_latency", "Mac M4 原生、FP32、8 步、每組 n=12（4 段文字×3 次）、斷音全部 0 次｜資料：results/03_streaming_bench.csv")

# ---------- 圖 3：速度 vs 品質 ----------
rows = {r["variant"]: r for r in read("results/05_quality.csv")}
show = [("fp32_s8", "FP32 8步（基準）"), ("fp32_s4", "FP32 4步"), ("fp32_s2", "FP32 2步"),
        ("int8ve_s8", "INT8 8步"), ("int8ve_s4", "INT8 4步"),
        ("int8ve_voc_keepends_s8", "INT8＋聲碼器保留頭尾"), ("int8all_s8", "INT8 全部（含聲碼器）")]
fig, ax = plt.subplots(figsize=(8, 4.6))
for key, name in show:
    r = rows[key]
    x, yv = float(r["rtf"]), float(r["WER_%"])
    broken = yv >= 20
    ax.scatter(x, yv, s=90, color=CRIT if broken else BLUE, marker="X" if broken else "o",
               edgecolor=SURF, linewidth=2, zorder=3)
    dy = {"int8ve_s8": 4, "int8ve_voc_keepends_s8": -5, "int8ve_s4": 4, "fp32_s4": 4,
          "fp32_s8": 4}.get(key, 3)
    ax.annotate(name + ("　← 壞掉" if broken else ""), (x, yv), xytext=(6, dy * 2),
                textcoords="offset points", fontsize=9.5, color=INK)
ax.set_xlabel("生成速度（生成耗時 ÷ 聲音長度，越小越快）")
ax.set_ylabel("錯字率 WER（%，whisper 轉回文字）")
ax.set_xlim(0.05, 0.42)
ax.set_ylim(-6, 95)
ax.set_title("INT8 只量化「逐步生成」：快 2.6 倍、錯字率仍 0%")
ax.legend(handles=[
    plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=9, label="可用"),
    plt.Line2D([], [], marker="X", ls="", color=CRIT, markersize=9, label="壞掉（錯字率 ≥ 20%）")],
    frameon=False, fontsize=9, loc="upper right")
save(fig, "3_speed_vs_quality", "Mac M4 原生、10 句車用語句｜資料：results/05_quality.csv")

# ---------- 圖 4：聲碼器哪幾層不能量化 ----------
rows = read("results/06_vocoder_sensitivity.csv")
names = {"keep_none": "全部量化", "keep_first1": "保留第 1 層", "keep_last1": "保留最後 1 層",
         "keep_last2_head": "保留最後 2 層\n（輸出頭）", "keep_first1_last2": "保留第 1 層\n＋最後 2 層"}
fig, ax = plt.subplots(figsize=(8, 3.8))
vals = [float(r["WER_%"]) for r in rows]
cols = [CRIT if v >= 20 else BLUE for v in vals]
ax.bar(range(len(rows)), vals, color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center", color=INK, fontsize=10)
ax.set_xticks(range(len(rows)), [names[r["case"]] for r in rows], fontsize=9.5)
ax.set_ylabel("錯字率 WER（%）")
ax.set_ylim(0, 100)
ax.set_title("聲碼器 INT8：關鍵在最後 2 層輸出頭")
ax.grid(axis="x", visible=False)
save(fig, "4_vocoder_sensitivity", "聲碼器共 33 層卷積；其餘層 INT8；逐步生成固定 INT8、8 步、10 句｜資料：results/06_vocoder_sensitivity.csv")

# ---------- 圖 5：資源受限（兩張並排，不共用 y 軸單位） ----------
rows = read("results/07_docker_cpu1_fp32.csv") + read("results/07_docker_cpu1_int8ve.csv")
g = defaultdict(list)
for r in rows:
    g[(r["variant"], float(r["prefill_s"]))].append(r)
pf = sorted({k[1] for k in g})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
for v, name, col, mk, ls in [("fp32", "FP32", ORANGE, "s", "--"), ("int8ve", "INT8", BLUE, "o", "-")]:
    fa = [S.median(float(x["first_audio_ms"]) for x in g[(v, p)]) / 1000 for p in pf]
    ur = [sum(float(x["underrun_ms"]) > 0 for x in g[(v, p)]) for p in pf]
    a1.plot(pf, fa, color=col, marker=mk, ls=ls, lw=2, markersize=8, label=name)
    a2.plot(pf, ur, color=col, marker=mk, ls=ls, lw=2, markersize=8, label=name)
    a1.annotate(name, (pf[-1], fa[-1]), xytext=(6, 0), textcoords="offset points", color=INK, va="center")
    a2.annotate(name, (pf[-1], ur[-1]), xytext=(6, 0), textcoords="offset points", color=INK, va="center")
n = len(g[("fp32", pf[0])])
a1.set_title("首段延遲（中位數）")
a1.set_xlabel("預填量（秒）")
a1.set_ylabel("秒")
a1.set_ylim(0, None)
a2.set_title(f"出現斷音的次數（共 {n} 次）")
a2.set_xlabel("預填量（秒）")
a2.set_ylabel("次")
a2.set_ylim(-0.3, n)
for a in (a1, a2):
    a.set_xticks(pf)
    a.legend(frameon=False, fontsize=9)
fig.suptitle("只剩 1 顆 CPU 時：加大預填只少一半斷音還更慢開口，INT8 直接歸零",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.0, 1, 0.95))
save(fig, "5_cpu_limited", "Docker --cpus=1（Mac 上的 Linux 虛擬機，非真實車用晶片）、單執行緒、8 步、逐句串流｜資料：results/07_docker_cpu1_*.csv")
