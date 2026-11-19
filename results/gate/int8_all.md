## 發版關卡：`int8_all` → ❌ 擋下

速度是 baseline 的 **3.44×**；機器：macOS-26.4-arm64-arm-64bit，10 核；耗時 168.6 秒

| 檢查 | 結果 | 門檻 | |
|---|---|---|---|
| 錄音辨識錯字率 | 82.0 % | ≤ 5.0 % | ❌ |
| 錯字率比 baseline 增加 | 82.0 pt | ≤ 2.0 pt | ❌ |
| 頻譜偏離（vs baseline） | 1.771  | ≤ 1.525（自然變化量）  | ❌ |
| 生成速度 RTF | 0.14 × | ≤ 0.482（baseline） × | ✅ |
| 峰值記憶體 | 254 MB | ≤ 573 MB | ✅ |
| 串流首段延遲 p50 | 0.241 s | ≤ 1.687 s | ✅ |
| 串流斷音 | 0 次 | ≤ 0 次 | ✅ |

聽錯的句子（前 5 句）：

- 原文「Turn left.」→ 聽成「Thank you.」
- 原文「In two hundred meters, turn right onto Main Street.」→ 聽成「Thank you.」
- 原文「Your battery is at twenty percent.」→ 聽成「Thank you.」
- 原文「The nearest charging station is four kilometers ahead, on the right side of the highway.」→ 聽成「Thank you.」
- 原文「Traffic is heavy on the route ahead, so I found a faster way that saves about twelve minutes.」→ 聽成「Thank you.」
