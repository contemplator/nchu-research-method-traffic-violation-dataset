# 🎤 期末報告 PPT 簡報提綱與完整內容（可直接交給 ChatGPT 生成投影片）

> **使用方式**：您可以直接將本文件全文（或分頁內容）貼給 ChatGPT，並附上指令：  
> *「請根據以下研究報告提綱與內容，幫我整理成適合向同學口頭報告的 10~12 頁 PowerPoint 簡報（每頁包含標題、條列重點、建議放的圖表位置與講者口白），風格簡潔好懂、重點數據清楚。」*

---

## 📌 簡報基本資訊
* **專題題目**：基於 YOLOv7 深度學習之台灣市區交通設施與違規標誌即時偵測與消融研究
* **課程名稱**：國立中興大學 研究方法（Research Methodology）期末專案
* **核心亮點**：從「人機協作標註 152 張台灣街景」出發，透過三階段控制變因消融實驗與領域超參數調校，將獨立測試集 `mAP@0.5` 從 `0.146` 提升至 `0.352`（成長 **2.4 倍**）、實務目標召回率突破 **51.2%**。

---

## 📑 簡報架構（共 11 頁，適合 8~12 分鐘口頭報告）

### Slide 1：封面頁 (Title Slide)
* **主標題**：基於 YOLOv7 之台灣交通設施與標誌偵測與消融研究
* **副標題**：Taiwan Traffic Facility & Sign Detection with Domain-Specific Hyperparameter Tuning
* **報告資訊**：國立中興大學 研究方法期末專題 ｜ 組員名單 / 日期

---

### Slide 2：研究背景與動機 (Motivation & Problem Statement)
* **為什麼做這個題目？**
  * 台灣市區道路環境複雜（機車多、招牌林立、號誌與標線密集），行車輔助與交通執法極需精準的設施與標誌辨識。
  * **台灣特有交通設施挑戰**：除了常見的紅綠燈 (`traffic_light`)，還有台灣特有的「機車兩段式左轉待轉區 (`two_stage_left_turn_box`)」與長條扁平的「行人穿越道 (`crosswalk`)」。
* **研究目標**：
  1. 建置高品質、貼近台灣真實街景的 6 類別交通設施資料集（152 張影像）。
  2. 探討 **模型骨幹大小**、**訓練輪數** 與 **超參數設計** 對偵測效能的影響（控制變因法）。
  3. 透過系統性誤差診斷（TP / FN / FP / F1 Curve），找出實務部署最佳信心門檻與模型極限。

---

### Slide 3：資料集建置與人機協作標註流程 (Dataset & Human-in-the-Loop Annotation)
* **資料規模與切分**：
  * 共 **152 張** 台灣街景影像（`1920×1080`），依 `7 : 2 : 1` 切分為 **Train 106 張 / Val 30 張 / Test 16 張**（測試集含 84 個真值目標）。
* **6 大標註類別與長尾分佈（Class Imbalance）**：
  * 多數類：`traffic_light` (393)、`crosswalk` (150)、`Sign_prohibit` (131)、`Sign_follow` (118)
  * 少數類：`two_stage_left_turn_box` (32)、`Sign_limit` (13) —— *埋下後續細分類混淆伏筆！*
* **標註實務痛點與創新解法（人機協作 Human-in-the-Loop）**：
  * **尺度取捨**：原圖縮小至 `640×640` 時，過遠的極小目標會模糊失真，因此過濾掉肉眼難辨的極遠目標，確保標註品質。
  * **加速標註**：純人工標註 30 張需耗時 30~40 分鐘；改採 **「人工示範 ➔ AI Agent 預標 50 張 ➔ 人工修正 ➔ AI Agent 二輪學習再標 50 張 ➔ 終審」**，大幅提升標註效率與一致性。

---

### Slide 4：實驗設計與環境挑戰排錯 (Methodology & Engineering Troubleshooting)
* **四組核心控制變因實驗（Ablation Study）**：
  * **Exp 1 (`YOLOv7-tiny`, 50 ep, 預設 hyp)**：輕量基準線（Baseline）。
  * **Exp 2 (`YOLOv7 Base`, 50 ep, 預設 hyp)**：對比 Exp 1，探討 **「模型大小 (6M vs 37.2M)」** 差異。
  * **Exp 3 (`YOLOv7-tiny`, 100 ep, 預設 hyp)**：對比 Exp 1，探討 **「訓練輪數 (50 vs 100)」** 差異。
  * **Exp 4 (三階段演進)**：探討 **「超參數調校」** 與 **「大型骨幹結合調優參數」** 的極限效能。
* **Colab 實作工程排錯（展現解決問題能力）**：
  * 解決 **NumPy 2.x** 移除 `np.trapz` 報錯、**PyTorch 2.6** `weights_only` 安全性阻擋，以及 `utils/google_utils.py` 觸發 GitHub API 限流崩潰與 `--exist-ok` 檔案覆寫殘留問題。

---

### Slide 5：問題診斷 — 為什麼初版超參數調校反而變差？(Stage 0 ➔ Stage 1)
* **負向調參現象**：
  * 初版 `Exp 4 (runs_origin)` 測試集 `mAP@0.5` 僅 `0.146`，甚至輸給未調參的 `Exp 3`（Val `0.161` vs `0.206`），漏檢率高達 **72.6%**！
* **三大根因診斷與領域知識修正 (`hyp.scratch.custom.yaml`)**：
  1. **學習率過小 (`lr0: 0.005 ➔ 0.01`)**：小資料集每輪步數少，減半學習率導致欠擬合。
  2. **幾何增強破壞扁平標線 (`degrees: 5.0 / shear: 2.0 ➔ 0.0`, `anchor_t: 4.0 ➔ 6.0`)**：斑馬線與待轉區為極扁平水平框，旋轉與剪切會使外接框膨脹混入柏油路雜訊；放寬 `anchor_t` 讓扁平框匹配更多正樣本。
  3. **模型過度保守 (`obj_pw: 1.0 ➔ 1.5`, `cls_pw: 1.0 ➔ 1.2`, 關閉 `mixup`)**：提高正樣本損失權重，改善高精確率卻大量漏抓的問題。
* **Stage 1 修正成效**：`YOLOv7-tiny` 測試集 `mAP@0.5` 由 `0.146` 躍升至 **`0.223` (+52.7%)**，實務抓到目標從 16 個增至 **26 個**，並成功框出斑馬線！

---

### Slide 6：模型架構與演進總對決 (Stage 1 ➔ Stage 2 ➔ Stage 3)
* **放上核心對比總表**（講解每一輪調整帶來的好與壞）：

| 實驗階段 / 模式 | 模型架構 | 輪數 | 超參數 | Val mAP@0.5 | Test mAP@0.5 | 效果評估 |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Exp 1** | `YOLOv7-tiny` | 50 | 官方預設 | `0.116` | - | ❌ 尚未收斂 |
| **Exp 3** | `YOLOv7-tiny` | 100 | 官方預設 | `0.206` | - | ⭕ 輪數加倍有效 |
| **Exp 4 初版** | `YOLOv7-tiny` | 100 | 初版自訂 | `0.161` | `0.146` | ❌ 負向調參 |
| **Exp 4 調優 (Tiny)** | `YOLOv7-tiny` | 100 | 領域調優 | `0.228` | `0.223` | ✅ 調參成功翻轉 |
| **Exp 2 (Base 50輪)** | `YOLOv7 Base` | 50 | 官方預設 | `0.244` | `0.280` | ✅ 大骨幹大幅降誤判 |
| **Exp 4 終極版 (Base)** | **`YOLOv7 Base`** | **100** | **領域調優** | **`~0.415`** | **`0.352`** | 🏆 **全面最佳 (成長 2.4 倍)** |

* **關鍵結論**：將驗證有效的「領域超參數」遷移至大型骨幹 `YOLOv7 Base` 並訓練 100 輪，產生 **1 + 1 > 2** 的協同效應！

---

### Slide 7：終極模型細部指標與類別突破 (Detailed Metrics of Final Model)
* **建議放圖**：[Exp 4 Base PR_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp4_base/test/exp4_test_evaluation/PR_curve.png) 與 [confusion_matrix.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp4_base/test/exp4_test_evaluation/confusion_matrix.png)
* **三大主力類別 AP@0.5 全面突破 0.50**：
  * `traffic_light`：**`0.689`**（測試集混淆矩陣檢出率 **77%**）
  * `Sign_follow`：**`0.634`**（測試集混淆矩陣檢出率 **79%**）
  * `Sign_prohibit`：**`0.502`**（測試集混淆矩陣檢出率 **61%**）
* **極困難少數類別突破**：
  * `two_stage_left_turn_box`（待轉區，全資料集僅 32 個）：在先前模型均為 0，終極版於測試集 R-Curve 首次突破至 **28.6% Recall (`AP=0.019`)**！

---

### Slide 8：可解釋性發現 — F1 Curve 與黃金信心門檻 (`conf = 0.40`)
* **建議放圖**：[runs_origin F1_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/test/exp4_test_evaluation/F1_curve.png) vs. [runs_exp4_base F1_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp4_base/test/exp4_test_evaluation/F1_curve.png)
* **現象解釋（Precision–Recall Trade-off）**：
  * 橫軸為信心門檻，曲線呈「兩端低、中間高」：門檻太低會亂猜（Precision≈0），門檻太高會全漏抓（Recall≈0）。
  * 調高 `obj_pw=1.5` 與充分訓練後，最佳 F1 峰頂由 `0.16 (@0.052)` 躍升並右移至 **`0.32 (@0.403)`**。
* **實務推論優化驗證（測試集 84 個目標）**：
  * **`conf >= 0.25`**：抓到 **43 個目標 (51.2% Recall)**，但含 53 個低信心誤判框 (FP)。
  * **改用最佳點 `conf >= 0.40`**：仍抓到 **41 個目標 (48.8% Recall)**，卻直接濾掉 **18 個背景誤判**（精確率升至 **`53.9%`**，實務 F1 達 **`51.2%`**）。

---

### Slide 9：實務推論視覺化展示與誤差診斷 (Visualizations & Error Analysis)
* **建議放圖**：[test_batch0_labels.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp4_base/test/exp4_test_evaluation/test_batch0_labels.jpg) vs. [test_batch0_pred.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp4_base/test/exp4_test_evaluation/test_batch0_pred.jpg)（或單張大圖 `KvNLWsBQYPE_000004`、`ZKJjA2sFqTE_000028`）
* **亮點案例**：
  * `KvNLWsBQYPE_000004`（12 個目標成功抓到 **8 個**，信心度達 `0.79~0.93`）。
  * `ZKJjA2sFqTE_000028`（12 個目標抓到 **7 個**，全場類別混淆從 7 個降至僅剩 2 個）。
* **誤差歸因分析（Error Analysis）**：
  1. **長尾效應造成的細分類混淆**：在 `000188`、`000229` 兩張圖中，模型已 **100% 精準定位到限速標誌 (`IoU >= 0.5`)**，但因訓練集 `Sign_prohibit` (89) 與 `Sign_limit` (8) 達 **11:1 不平衡**且同為紅圈外觀，導致被分類為 `Sign_prohibit`。
  2. **透視與局部遮擋**：斑馬線面積大且常被車流遮擋，部分預測框雖有框出但 `IoU` 落在 `0.3~0.48`，未達 `0.5` 嚴格計分門檻。

---

### Slide 10：研究結論 (Conclusions)
1. **標註與資料工程**：人機協作（AI Agent 預標 + 人工修訂）能有效降低標註成本，同時過濾極遠模糊目標有助於訓練穩定。
2. **領域知識調參的重要性**：通用型資料增強（旋轉/剪切）不適用於道路水平標線；針對性修正 `lr0`、`anchor_t` 與正樣本權重使輕量模型測試 mAP 提升 **52.7%**。
3. **骨幹容量與閾值調校**：`YOLOv7 Base + 100 輪 + 調優超參數` 達到專案最高 **`Test mAP@0.5 = 0.352`**（成長 2.4 倍），結合 `F1 Curve` 選定 `conf=0.40` 可達成最佳實務精準/召回平衡。

---

### Slide 11：研究限制與未來展望 (Limitations & Future Work)
* **資料集規模與不平衡**：針對 `Sign_limit`（13 個）與 `two_stage_left_turn_box`（32 個）擴充台灣不同縣市、夜間與雨天場景樣本。
* **進階損失函數**：針對難分類之相似圓牌與極端稀缺類別，測試啟用 **Focal Loss (`fl_gamma > 0`)** 或 **類別加權採樣 (Class-Balanced Sampling)**。
* **即時部署評估**：比較 `YOLOv7-tiny` 與 `YOLOv7 Base` 在邊緣裝置上的 FPS 推論速度，評估實際車載系統之精度–速度權衡。

---

### Slide 12：Q & A（感謝聆聽）
* 附上專案 GitHub Repository 與 Colab 實驗連結，歡迎提問與指教！
