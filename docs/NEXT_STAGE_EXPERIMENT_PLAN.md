# 📋 下階段實驗計劃書：458 張完整資料集與 Append-Only 消融實驗架構 (Exp 1~9)

> **用途說明**：本文件整合了原專題（152 張基準研究）與《胸部 X 光 YOLO_Study》報告的雙方長處，作為下一階段切換至新 Git 分支與新對話時的完整執行藍圖。

---

## 一、 專案現況與資料集基準（已完成前處理）

1. **完整標註資料集（458 張影像）**：
   - `data/annotated/`：458 張 `.jpg` 原始影像、457 個 `.json`（其中 `ZKJjA2sFqTE_000378_t00378.jpg` 為無目標負樣本，僅有空白 `.txt`）。
   - `data/labels/`：458 個 YOLO `.txt` 標註檔（已統一修正組員 `wupin` 原本錯位的類別順序，並通過全部 457 組 JSON/TXT 交叉驗證，0 筆錯誤）。
   - **標準 6 類別順序**：
     - `0: traffic_light`（1,212 框）
     - `1: crosswalk`（442 框）
     - `2: two_stage_left_turn_box`（76 框）
     - `3: Sign_follow`（397 框）
     - `4: Sign_prohibit`（488 框）
     - `5: Sign_limit`（37 框）
2. **切分與打包成果**：
   - 已透過 `scripts/split_dataset.py` 依 `7 : 2 : 1`（`seed=42`）切分至 `data/dataset/` 並打包為 `data/dataset.zip`（197.80 MB）：
     - **Train**：320 張
     - **Val**：91 張
     - **Test**：47 張（獨立保留測試集）

---

## 二、 融合雙方報告長處之核心設計原則

結合我們原本「人機協作標註 + 負向調參診斷 + F1 門檻折衷」的優勢，並借鏡《胸部 X 光 YOLO_Study》的嚴謹實驗方法，下階段強制執行以下規範：

1. **Append-Only 實驗登錄表（保留歷史設定、只增不改）**：
   - **完全保留 `Exp 1 ~ Exp 4` 的原始設定**（包含原本的 `50 / 100` Epochs、`batch-size`、`cfg` 與 `hyp`），確保舊實驗結果在任何時間點都能 100% 重現，並在同一個 Git commit 中完整保留所有實驗脈絡。
   - 後續新策略一律以 **`Exp 5, Exp 6, Exp 7, Exp 8, Exp 9 ...`** 往後累加，不覆寫既有實驗編號。
   - 在 `NCHU_Research_YOLOv7.ipynb` 中以結構化清單 `EXPERIMENTS` 管理，每組實驗皆提供 `"enabled": True / False` 開關，迭代新實驗時可自由跳過已跑完的組別，同時在程式碼中保留所有歷史設定。
2. **純淨控制變因對照**：
   - 所有實驗固定 **Seed = 42**、同一份 **320 / 91 / 47** 資料切分。
   - 延續既有 `Exp 2 / Exp 4` 的設定，所有 `YOLOv7 Base` 實驗（`Exp 2, 4, 5, 7, 8, 9`）統一維持 **`batch-size = 8`**，使 `Exp 4 (640)` 與 `Exp 7 (960)` 之間成為百分百純淨的「單一解析度變因」對照。
3. **嚴格防資料洩漏（Validation-First Thresholding）**：
   - 先在 **Val 驗證集（91 張）** 上透過 F1 Curve 選出各實驗的最佳信心門檻 `best_conf`，再統一套用至 **Test 測試集（47 張）** 進行最終盲測評估。
4. **四道資料品質防線與負樣本論述**：
   - 在 Colab Step 4 解壓 `dataset.zip` 後，自動執行 `scripts/verify_annotations.py` 檢查檔案完整度、類別分佈與負樣本比例（說明無目標負樣本有助於抑制背景假陽性 FP）。
5. **量化與視覺化對齊升級**：
   - 自動產出 **Master Benchmark 總對照表**（含 `mAP@0.5`、`mAP@0.5:0.95`、`Precision`、`Recall`、`F1`、推論延遲 `ms/img` & `FPS`）。
   - 自動繪製 **4 象限收斂曲線圖**（`mAP@0.5`、`mAP@0.5:0.95`、`Total Train Loss`、`Recall`）。
   - 自動統計 **4 維度誤差分類學**（`FP` 假陽性、`FN` 假陰性、`Misclassification` 類別混淆、`Low IoU (0.2 ≤ IoU < 0.5)` 邊界偏差）並產出 **所有啟用實驗的同圖並排推論對比矩陣（`inference_comparison_grid.jpg`）**。

---

## 三、 下階段 `Exp 1 ~ Exp 9` 實驗矩陣規劃（保留 Exp 1~4 + 累加 Exp 5~9）

### 1. 保留既有基準實驗（原封不動）
| 實驗組別 | 模型架構 | 解析度 | Batch | Epochs | 超參數設定檔 | 保留目的與角色 |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Exp 1** | `YOLOv7-tiny` | `640` | 16 | 50 | `data/hyp.scratch.tiny.yaml` | 既有輕量 50 輪基準線（`exp1_yolov7_tiny_baseline`）。 |
| **Exp 2** | `YOLOv7 Base` | `640` | 8 | 50 | `data/hyp.scratch.p5.yaml` | 既有大骨幹 50 輪基準（`exp2_yolov7_base_scale`）。 |
| **Exp 3** | `YOLOv7-tiny` | `640` | 16 | 100 | `data/hyp.scratch.tiny.yaml` | 既有輕量 100 輪基準（`exp3_yolov7_tiny_100ep`）。 |
| **Exp 4** | `YOLOv7 Base` | `640` | 8 | 100 | `configs/hyp.scratch.custom.yaml` | 既有大骨幹 + 領域調優 v1（`exp4_yolov7_tiny_tuned` 原名稱保留以相容舊路徑）。 |

### 2. 新增進階對照實驗（往後累加）
| 實驗組別 | 模型架構 | 解析度 | Batch | Epochs | 超參數設定檔 | 核心對照關係與驗證假說 |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Exp 5** | `YOLOv7 Base` | `640` | 8 | 100 | `data/hyp.scratch.p5.yaml` | **補齊公平對照（Base + 100ep + 預設參數）**：<br>• 對比 `Exp 2`：驗證 Base 模型從 50 ➔ 100 輪的輪數增益。<br>• 對比 `Exp 4`：在同為 100 輪、Base、`batch=8` 下，驗證領域參數 v1 的純淨增益。 |
| **Exp 6** | `YOLOv7-tiny` | `640` | 16 | 100 | `configs/hyp.scratch.custom.yaml` | **補齊公平對照（Tiny + 100ep + 領域調優 v1）**：<br>• 對比 `Exp 3`：在 458 張完整資料集上量測領域調優對輕量模型的純增益與邊緣 FPS。 |
| **Exp 7** | `YOLOv7 Base` | **`960`** | 8 | 100 | 領域調優 v1 高解析度版 (`configs/hyp.traffic_highres.yaml`) | **對比 Exp 4（借鏡 X 光高解析度策略）**：維持 `batch=8`、`100ep`，僅將輸入升至 `960×960`（並微調 `lr0=0.005` 防止大特徵圖震盪），驗證遠距紅綠燈與小標誌細節保留效益。 |
| **Exp 8** | `YOLOv7 Base` | **`960`** | 8 | 100 | **領域進階版 v2 (`configs/hyp.traffic_v2.yaml`)** | **對比 Exp 7（長尾分類 + 標線定位精修）**：針對先前兩大殘留誤差設計：<br>1. 啟用 **Focal Loss (`fl_gamma: 1.0`)**，改善 `Sign_prohibit` vs `Sign_limit` (11:1 長尾混淆) 與 `two_stage_left_turn_box`。<br>2. 關閉 **`fliplr: 0.0`** 保護方向性標誌語意。<br>3. 降 **`mosaic: 0.5`** 減少拼接裁切對長條 `crosswalk` 的破壞。<br>4. 升 **`box: 0.075`** 提高邊界框回歸權重，降低 `Low IoU` 誤差。 |
| **Exp 9** | `YOLOv7 Base` | **`960`** | 8 | 100 | 領域進階版 v2 (`configs/hyp.traffic_v2.yaml`) + **CLAHE 影像前處理** | **對比 Exp 8（直接引入 X 光 CLAHE 策略應對夜間／陰影）**：維持與 `Exp 8` 完全相同的架構與超參數，僅對資料集（Train / Val / Test）的 **LAB 色彩空間 `L` 通道（或 HSV `V` 通道）套用自適應直方圖均衡化 `CLAHE (clipLimit=2.0, tileGridSize=(8,8))`** 產出增強版影像副本（不覆蓋原圖，使用專屬 `data_clahe.yaml`），驗證局部對比度增強是否能拉高逆光、樹蔭下斑馬線與夜間標誌的召回率。 |

> **註**：在 `EXPERIMENTS` 清單中，第一次於 458 張資料集跑完整流程可全開 `Exp 1 ~ Exp 9`（或先開 `Exp 1 ~ 8`，再單獨啟用 `Exp 9`）；若時間有限或後續只想測試新策略，只需將已跑完的實驗設為 `"enabled": False` 即可直接跳過，設定仍完整保留在檔案與 commit 中。

---

## 四、 新分支實作修改清單（Checklist）

切換至新 Git 分支後，依序修改以下檔案：

1. **新增超參數設定檔（`configs/`）**：
   - `configs/hyp.traffic_highres.yaml`（以 `hyp.scratch.custom.yaml` 為基礎，將 `lr0` 調為 `0.005` 供 Exp 7 使用）。
   - `configs/hyp.traffic_v2.yaml`（以 `hyp.traffic_highres.yaml` 為基礎，設定 `fl_gamma: 1.0`、`fliplr: 0.0`、`mosaic: 0.5`、`box: 0.075` 供 Exp 8 使用）。
2. **新增前處理腳本與升級分析腳本（`scripts/`）**：
   - 新增 `scripts/preprocess_clahe.py`：針對 `data/dataset/` 的圖片於 `LAB` 空間 `L` 通道執行 `CLAHE (clipLimit=2.0, tileGridSize=(8,8))` 並輸出至獨立目錄 `data/dataset_clahe/` 與 `configs/traffic_facility_clahe.yaml`（供 Exp 9 使用，不影響 Exp 1~8 原始影像）。
   - 確認 `scripts/verify_annotations.py` 可直接在 Colab Step 4 執行並輸出各 split 類別與負樣本統計。
   - 擴充 `scripts/analyze_test_results.py`：
     - 支援動態讀取所有啟用實驗（Exp 1 ~ Exp 9+）的 `results.txt` 並繪製 **4 象限收斂對照圖**。
     - 支援對 47 張 Test Set 計算各實驗的 **4 維度誤差統計表**（`TP`、`FP`、`FN`、`Misclassification`、`Low IoU`）。
     - 支援自動產出 **Side-by-Side 橫向推論對比矩陣圖**（挑選含 `Sign_limit`、`two_stage_left_turn_box`、`crosswalk` 等難例）。
3. **重構 `notebooks/NCHU_Research_YOLOv7.ipynb`**：
   - **Step 4**：解壓 `data/dataset.zip` 後執行資料品質防線驗證。
   - **Step 5**：改為結構化 `EXPERIMENTS = [...]` 清單與迴圈（含 `"enabled": True/False` 跳過機制），完整保留 Exp 1~4 原設定並累加 Exp 5~9；每組訓練結束後立即備份權重與日誌至 Google Drive（防止 Colab 斷線遺失進度）。
   - **Step 6**：先於 `val` 集找出最佳 F1 信心門檻，再統一於 `test` 集執行獨立評估並記錄推論速度（`ms/img` 與 `FPS`）。
   - **Step 7 & 8**：呼叫升級後的分析腳本，自動產出總對照表、4 象限收斂圖、4 維度誤差表與橫向推論對比矩陣。
   - **Step 9**：將完整實驗產出打包備份至 Google Drive。

---

## 五、 下階段報告與簡報調整重點

完成 Exp 1 ~ Exp 9 實驗後，建議同步更新 `docs/PPT_OUTLINE_AND_CONTENT.md` 與期末簡報架構：
- **資料集頁**：更新為 **458 張（320 / 91 / 47）** 與四道品質驗證結果，並說明空白負樣本抑制 FP 的角色。
- **實驗設計頁**：放入嚴謹的「控制變因 vs 自變數」對照表與 Exp 1 ~ Exp 9 階梯圖。
- **量化分析頁**：放入 **總對照表**、**4 象限收斂曲線** 與 **4 維度誤差分類學降幅統計**。
- **視覺化對比頁**：放入 **同圖並排推論矩陣**，直觀呈現從 `Tiny-640` ➔ `Base-640` ➔ `Base-960-v2` ➔ `Base-960-CLAHE` 的演進。
- **實務部署頁**：結合 `Exp 3 / Exp 6 (Tiny)` 與 `Exp 7 / Exp 8 / Exp 9 (Base-960)` 的 FPS 與 mAP 數據，提出「邊緣端即時預警 vs 雲端伺服器高精度稽核」之算力與延遲權衡建議。
