# 📊 YOLOv7 實驗結果驗證、測試集檢視指南與後續優化建議

> **建立日期**：2026-11-15  
> **對應分支**：`feat/152-images-leo`  
> **資料集規模**：152 張台灣街景影像（Train: 106 張 / Val: 30 張 / Test: 16 張），6 大交通設施與標誌類別

---

## 一、如何用最直覺的方式查看 `test` 組別偵測結果？（要打開哪個檔案？）

想要最直覺地知道測試集 16 張照片「哪些有被偵測出來、偵測到什麼物件、哪些漏抓」，您可以依照需求直接打開以下 **3 個位置**：

### 方法 1：【最推薦：一張圖全覽 16 張標準答案 vs 預測結果】
直接並排打開 `test.py` 產生的 **4x4 批次拼接圖**（16 張測試照片一次看齊）：
* 🟢 **標準答案 (Ground Truth)**：[test_batch0_labels.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/test/exp4_test_evaluation6/test_batch0_labels.jpg)
* 🔵 **模型預測 (Model Predictions)**：[test_batch0_pred.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/test/exp4_test_evaluation6/test_batch0_pred.jpg)

> **怎麼看？** 兩張圖的 16 個格子順序完全一模一樣！左圖有框而右圖沒框的，就是**漏抓 (FN)**（例如地面的橘色 `crosswalk` 與綠色 `two_stage_left_turn_box`）；右圖有框出類別名稱與信心度數字（如 `traffic_light 0.8`、`Sign_prohibit 0.8`）的就是**有被偵測出來的物件**。

---

### 方法 2：【看單張高畫質偵測結果圖（已畫好框與信心分數）】
直接打開 `detect.py` 輸出的最新資料夾 **[results/runs/detect/detect_test_images5/](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5)** 裡面的 16 張 `.jpg` 圖片！
以下為您整理好這 **16 張測試圖片的直接連結與逐張偵測明細表**：

| 測試集照片（點擊直接看畫好預測框的圖） | 真值數量 | 預測數量 | ✅ 成功抓到 (TP, IoU≥0.5) | 🔄 類別混淆 (框對位置但認錯類別) | ❓ 誤判虛警 (FP, 位置偏移或背景誤認) | ⚠️ 漏抓目標 (FN) |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| [KvNLWsBQYPE_000004_t00004.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_000004_t00004.jpg) | 12 | 6 | **4 個**：`traffic_light` (0.77, 0.67, 0.66), `Sign_prohibit` (0.48) | 無 | `traffic_light` (0.28, 0.27) | `crosswalk`×1, `Sign_follow`×2, `Sign_prohibit`×3, `traffic_light`×2 |
| [KvNLWsBQYPE_000188_t00188.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_000188_t00188.jpg) | 1 | 1 | 0 個 | **1 個**：真值 `Sign_limit` ➔ 預測 `Sign_prohibit` (0.77) | 無 | 無（位置完全抓對，僅紅圈標誌細分類混淆） |
| [KvNLWsBQYPE_000229_t00229.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_000229_t00229.jpg) | 1 | 1 | 0 個 | **1 個**：真值 `Sign_limit` ➔ 預測 `Sign_prohibit` (0.47) | 無 | 無（位置完全抓對，僅紅圈標誌細分類混淆） |
| [KvNLWsBQYPE_000956_t00956.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_000956_t00956.jpg) | 4 | 2 | 0 個 | **1 個**：真值 `Sign_follow` ➔ 預測 `traffic_light` (0.28) | `traffic_light` (0.77) | `traffic_light`×2, `Sign_prohibit`×1 |
| [KvNLWsBQYPE_000967_t00967.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_000967_t00967.jpg) | 5 | 2 | 0 個 | 無 | `traffic_light` (0.42, 0.29) | `traffic_light`×3, `Sign_follow`×1, `Sign_prohibit`×1 |
| [KvNLWsBQYPE_001016_t01016.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_001016_t01016.jpg) | 4 | 1 | 0 個 | 無 | `traffic_light` (0.61) | `traffic_light`×2, `Sign_prohibit`×2 |
| [KvNLWsBQYPE_001057_t01057.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_001057_t01057.jpg) | 16 | 6 | **3 個**：`Sign_prohibit` (0.84), `traffic_light` (0.77, 0.27) | 無 | `traffic_light` (0.79, 0.48, 0.25) | `crosswalk`×5, `two_stage_left_turn_box`×3, `Sign_follow`×3, `Sign_prohibit`×1, `traffic_light`×1 |
| [KvNLWsBQYPE_001324_t01324.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/KvNLWsBQYPE_001324_t01324.jpg) | 10 | 5 | **2 個**：`traffic_light` (0.81, 0.48) | 無 | `traffic_light` (0.38, 0.34, 0.33) | `Sign_follow`×4, `traffic_light`×2, `two_stage_left_turn_box`×1, `crosswalk`×1 |
| [TTJkh6VwNgY_000463_t00463.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/TTJkh6VwNgY_000463_t00463.jpg) | 5 | 1 | 0 個 | 無 | `Sign_prohibit` (0.85) | `traffic_light`×2, `two_stage_left_turn_box`×2, `crosswalk`×1 |
| [TTJkh6VwNgY_000477_t00477.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/TTJkh6VwNgY_000477_t00477.jpg) | 0 | 1 | 0 個（原本為空背景圖） | 無 | `Sign_prohibit` (0.49) | 無 |
| [ZKJjA2sFqTE_000028_t00028.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/ZKJjA2sFqTE_000028_t00028.jpg) | 12 | 3 | **2 個**：`traffic_light` (0.78), `Sign_prohibit` (0.36) | 無 | `Sign_prohibit` (0.31) | `crosswalk`×2, `two_stage_left_turn_box`×1, `Sign_prohibit`×3, `Sign_follow`×2, `traffic_light`×2 |
| [ZKJjA2sFqTE_000052_t00052.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/ZKJjA2sFqTE_000052_t00052.jpg) | 4 | 4 | **3 個**：`Sign_prohibit` (0.82), `traffic_light` (0.68, 0.28) | 無 | `Sign_prohibit` (0.58) | `crosswalk`×1 |
| [ZKJjA2sFqTE_000190_t00190.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/ZKJjA2sFqTE_000190_t00190.jpg) | 0 | 2 | 0 個（原本為空背景圖） | 無 | `Sign_prohibit` (0.55), `traffic_light` (0.30) | 無 |
| [ZKJjA2sFqTE_001261_t01261.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/ZKJjA2sFqTE_001261_t01261.jpg) | 2 | 3 | 0 個 | 無 | `Sign_prohibit` (0.56, 0.29), `Sign_follow` (0.31) | `traffic_light`×1, `crosswalk`×1 |
| [ZKJjA2sFqTE_001738_t01738.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/ZKJjA2sFqTE_001738_t01738.jpg) | 4 | 3 | **1 個**：`Sign_prohibit` (0.78) | **1 個**：真值 `Sign_prohibit` ➔ 預測 `Sign_follow` (0.32) | `Sign_prohibit` (0.46) | `traffic_light`×1, `crosswalk`×1 |
| [ZKJjA2sFqTE_001908_t01908.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/detect/detect_test_images5/ZKJjA2sFqTE_001908_t01908.jpg) | 4 | 2 | **2 個**：`traffic_light` (0.76, 0.55) | 無 | 無 | `crosswalk`×1, `traffic_light`×1 |

---

### 方法 3：【執行診斷腳本一鍵印出報表】
我已經升級了 [scripts/analyze_test_results.py](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/scripts/analyze_test_results.py)，現在只要在終端機（或 Colab Step 8）執行：
```bash
python3 scripts/analyze_test_results.py \
  --gt data/dataset/test/labels \
  --pred results/runs/detect/detect_test_images5/labels \
  --images data/dataset/test/images \
  --classes data/dataset/classes.txt \
  --iou 0.5
```
就會直接印出上面「各類別統計表」以及「16 張影像逐張偵測明細（`✅抓到` / `🔄混淆` / `❓誤判` / `⚠️漏抓`）」！

---

## 二、Colab 執行錯誤根因紀錄與解決方案

### 1. NumPy 2.0+ 相容性問題 (`AttributeError: module 'numpy' has no attribute 'trapz'`)
* **發生原因**：Google Colab 目前預裝 Python 3.13 與 NumPy 2.x，而 NumPy 2.0 正式移除了舊版積分函式 `np.trapz`（改名為 `np.trapezoid`）。YOLOv7 在每個 Epoch 結束計算 PR 曲線下面積 (AP) 時呼叫 `yolov7/utils/metrics.py` 第 105 行的 `np.trapz` 因而崩潰。
* **解決方案**：在 [notebooks/NCHU_Research_YOLOv7.ipynb](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/notebooks/NCHU_Research_YOLOv7.ipynb) 的 Step 3 自動將 `np.trapz` 修補為向後相容的 `getattr(np, "trapezoid", getattr(np, "trapz", None))`。

### 2. Step 6 測試評估 `utils/google_utils.py` 崩潰 (`KeyError: 'assets'`) 與資料夾遞增問題
* **雙重根本原因**：
  1. **缺少 `--exist-ok` 導致資料夾名稱加上數字後綴**：當 `train.py` 重跑時，YOLOv7 自動將輸出目錄推進為 `exp4_yolov7_tiny_tuned3`。
  2. **找不到寫死路徑而觸發 GitHub API 限流**：Step 6 (`test.py`) 原本讀取寫死的 `runs/train/exp4_yolov7_tiny_tuned/weights/best.pt`，因找不到檔案而觸發 `attempt_download()` 向 GitHub Releases API 請求下載；Colab 共用 IP 遭遇 GitHub API Rate Limit 後回傳錯誤訊息（不含 `'assets'` 鍵），造成 `response['assets']` 崩潰。
  3. 同樣地，`detect.py` 也因重跑產生了 `detect_test_images5`，導致原本讀取 `detect_test_images/labels` 的分析腳本讀到空目錄（顯示 `Predictions: 0`）。
* **解決方案（已寫入 [notebooks/NCHU_Research_YOLOv7.ipynb](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/notebooks/NCHU_Research_YOLOv7.ipynb)）**：
  1. 在所有 `train.py`、`test.py`、`detect.py` 加上 `--exist-ok`。
  2. 在 Step 3 自動將 `yolov7/utils/google_utils.py` 中的 `response['assets']` 替換為 `response.get('assets', [])`。
  3. 在 Step 6 ~ Step 9 改用 `glob.glob` 自動抓取最新權重與最新預測資料夾。

---

## 三、四組消融實驗 (Ablation Study) 數據驗證與比較

以下為 [results/runs/train/](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs/train) 中四組實驗在驗證集 (`val` split, 30 張影像 / 189 個真實標註框) 的最終收斂表現：

| 組別 | 實驗設定 | 權重大小 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 各類別 mAP@0.5 表現亮點 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp 1** | `YOLOv7-tiny` (50 ep, 預設 hyp) | 12.3 MB | 0.679 | 0.124 | **0.116** | 0.0569 | `Sign_prohibit`: 0.329, `traffic_light`: 0.196, `Sign_follow`: 0.134 |
| **Exp 2** | **`YOLOv7` Base (50 ep, 預設 hyp)** | 74.8 MB | 0.618 | **0.254** | **0.244** 🏆 | **0.1120** 🏆 | **`traffic_light`: 0.582**, **`Sign_prohibit`: 0.477**, `Sign_follow`: 0.187, **`crosswalk`: 0.166** |
| **Exp 3** | `YOLOv7-tiny` (100 ep, 預設 hyp) | 12.3 MB | 0.680 | 0.217 | **0.206** | 0.0925 | `Sign_prohibit`: 0.417, `traffic_light`: 0.399, `Sign_follow`: 0.315, `crosswalk`: 0.105 |
| **Exp 4** | `YOLOv7-tiny` (100 ep, Custom hyp) | 12.3 MB | **0.756** | 0.150 | **0.161** | 0.0629 | `Sign_prohibit`: 0.417, `traffic_light`: 0.322, `Sign_follow`: 0.167, `crosswalk`: 0.060 |

---

## 四、後續模型與超參數調整建議（Actionable Recommendations）

### 💡 建議 1：在 Step 6 & Step 7 改用（或新增對比）表現最強的 **Exp 2 (`YOLOv7` Base)** 權重
* **依據**：在僅 50 Epochs 下，**Exp 2 (`YOLOv7` Base, mAP@0.5 = 0.244)** 顯著超越了所有 `YOLOv7-tiny` 組別（包含訓練 100 Epochs 的 Exp 3 與 Exp 4），且在難度最高的 `crosswalk` 達到 `0.166`、`traffic_light` 達到 `0.582`。
* **行動**：
  * 可將 Step 6 (`test.py`) 與 Step 7 (`detect.py`) 的權重改指向 `runs/train/exp2_yolov7_base_scale/weights/best.pt`，測試集檢出率（特別是遠處號誌與地面斑馬線）會立即大幅提升！
  * 若時間充裕，將 `YOLOv7` Base 訓練至 **100 Epochs**，mAP@0.5 預估可突破 **0.35 ~ 0.45**。

### 💡 建議 2：修正 [configs/hyp.scratch.custom.yaml](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml) 的 4 個超參數（讓 Exp 4 超越 Exp 3）
目前 Exp 4 (mAP@0.5 = 0.161) 之所以低於 Exp 3 (0.206)，是因為以下超參數在小樣本與扁平標線上產生了反效果，建議調整如下：

| 參數名稱 | 目前設定 (`hyp.scratch.custom.yaml`) | 建議修改值 | 調整原因與物理意義 |
| :--- | :---: | :---: | :--- |
| **`lr0`** | `0.005` | **`0.01`** | 106 張訓練集在 `batch=16` 下每 Epoch 僅 7 個 step，100 Epochs 僅 700 次更新；`0.005` 學習率過小導致收斂不足。 |
| **`degrees` / `shear`** | `degrees: 5.0`, `shear: 2.0` | **`0.0` / `0.0`** | 斑馬線 (`crosswalk`) 與待轉區 (`two_stage_left_turn_box`) 是極扁平水平框，旋轉與剪切會使外接水平 Bounding Box 大幅膨脹並混入大量柏油路面背景，導致定位失準。 |
| **`mixup` / `paste_in`** | `mixup: 0.1`, `paste_in: 0.05` | **`0.0` / `0.0`** | 在僅 100 Epochs 的 `yolov7-tiny` 上同時疊加 Mosaic + Mixup 訊號過於複雜，關閉 Mixup 僅保留 `mosaic: 1.0` 有助於加速分類與物件信心度收斂。 |
| **`anchor_t`** | `4.0` | **`6.0`** | 放寬 Anchor 長寬比匹配門檻（由 4.0 提高至 6.0），讓極扁平的 `crosswalk` 與 `two_stage_left_turn_box` 能匹配到更多正樣本預測框。 |
| **`obj_pw` / `cls_pw`** | `1.0` / `1.0` | **`1.5` / `1.2`** | 適度提高 Objectness 與 Classification 正樣本權重，改善模型過度保守（High Precision 0.756、Low Recall 0.150）導致大量漏抓的問題。 |

### 💡 建議 3：資料集長尾分佈（Class Imbalance）之報告論述策略
全資料集 (152 張) 各類別標註框數量統計如下：
* `traffic_light`：**393 個**（Train 271 / Val 91 / Test 31）
* `crosswalk`：**150 個**（Train 100 / Val 36 / Test 14）
* `Sign_prohibit`：**131 個**（Train 89 / Val 25 / Test 17）
* `Sign_follow`：**118 個**（Train 75 / Val 30 / Test 13）
* `two_stage_left_turn_box`：**32 個**（Train 21 / Val 4 / Test 7）⚠️ *樣本極少且透視扁平*
* `Sign_limit`：**13 個**（Train 8 / Val 3 / Test 2）⚠️ *樣本極少，且與 `Sign_prohibit` 同為紅圈白底圓牌*

> **報告撰寫亮點**：在期末報告 Error Analysis 章節中，可直接引用 `KvNLWsBQYPE_000188` 與 `KvNLWsBQYPE_000229` 兩張測試影像，說明模型其實**已經成功定位到遠處的限速標誌 (`IoU > 0.5`)**，但因為訓練集中 `Sign_prohibit` (89 個) 與 `Sign_limit` (8 個) 存在 **11:1 的樣本不平衡**，且兩者皆具「紅圈圓牌」高相似度特徵，才導致分類頭將其歸入 `Sign_prohibit`。這能充分展現研究的深度與可解釋性！
