# 🏆 模型架構與超參數調優測試集正面對決報告：`Exp 2 (YOLOv7 Base)` vs. `Exp 4 (YOLOv7-tiny Tuned)`

> **建立日期**：2026-11-17  
> **資料來源目錄**：[results/runs_exp2](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2)（已清除舊有殘留檔案之純淨重跑結果）  
> **對照組別**：
> * **模型 A（大型骨幹代表）**：`Exp 2` — `YOLOv7 Base` (`cfg/training/yolov7.yaml`, 37.2M 參數 / 74.8 MB)，50 Epochs，官方預設超參數 (`hyp.scratch.p5.yaml`)
> * **模型 B（輕量調優代表）**：`Exp 4` — `YOLOv7-tiny Tuned` (`configs/yolov7-tiny-custom.yaml`, 6.0M 參數 / 12.3 MB)，100 Epochs，領域微調超參數 ([configs/hyp.scratch.custom.yaml](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml))
> * **基準參考（調參前初版）**：`runs_origin Exp 4` — `YOLOv7-tiny`，100 Epochs，初版未最佳化超參數 ([configs/hyp.scratch.custom.yaml.original](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml.original))

---

## 一、四組消融實驗設定與驗證集 (Val Split, 30 張影像 / 189 個目標) 總覽

在進入獨立測試集 (`test` split) 比較前，下表彙整 `runs_exp2/train/` 中四組實驗的控制變因與驗證集收斂成果：

| 實驗組別 | 模型架構 | 權重大小 | 訓練輪數 (`Epochs`) | Batch Size | 超參數配置 (`--hyp`) | Val Precision | Val Recall | Val mAP@0.5 | Val mAP@0.5:0.95 | 訓練曲線與混淆矩陣 |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1** | `YOLOv7-tiny` | 12.3 MB | 50 | 16 | 官方預設 (`hyp.scratch.tiny.yaml`) | `0.679` | `0.124` | `0.116` | `0.0569` | [results.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp1_yolov7_tiny_baseline/results.png) / [CM](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp1_yolov7_tiny_baseline/confusion_matrix.png) |
| **Exp 2** | **`YOLOv7 Base`** | **74.8 MB** | **50** | 8 | 官方預設 (`hyp.scratch.p5.yaml`) | `0.618` | **`0.254`** | **`0.244`** 🏆 | **`0.1120`** 🏆 | [results.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp2_yolov7_base_scale/results.png) / [CM](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp2_yolov7_base_scale/confusion_matrix.png) |
| **Exp 3** | `YOLOv7-tiny` | 12.3 MB | 100 | 16 | 官方預設 (`hyp.scratch.tiny.yaml`) | `0.680` | `0.217` | `0.206` | `0.0925` | [results.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp3_yolov7_tiny_100ep/results.png) / [CM](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp3_yolov7_tiny_100ep/confusion_matrix.png) |
| **Exp 4 (初版)** | `YOLOv7-tiny` | 12.3 MB | 100 | 16 | 初版自訂 (`hyp.scratch.custom.yaml.original`) | `0.756` | `0.150` | `0.161` | `0.0629` | [results.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/train/exp4_yolov7_tiny_tuned/results.png) / [CM](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/train/exp4_yolov7_tiny_tuned/confusion_matrix.png) |
| **Exp 4 (純淨調優)** | **`YOLOv7-tiny`** | **12.3 MB** | **100** | 16 | **領域調優 ([hyp.scratch.custom.yaml](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml))** | **`~0.720`** | **`~0.220`** | **`~0.228`** 🥈 | **`~0.1030`** 🥈 | [results.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp4_yolov7_tiny_tuned/results.png) / [CM](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/train/exp4_yolov7_tiny_tuned/confusion_matrix.png) |

---

## 二、獨立測試集 (`test` split, 16 張影像 / 84 個目標) 指標全面對比 (`Step 6: test.py`)

以下數據取自 `runs_exp2/test/` 下之獨立測試集評估結果：
* **Exp 2 (`YOLOv7 Base`) 圖表**：[PR_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp2_test_evaluation/PR_curve.png) ｜ [F1_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp2_test_evaluation/F1_curve.png) ｜ [R_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp2_test_evaluation/R_curve.png) ｜ [confusion_matrix.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp2_test_evaluation/confusion_matrix.png) ｜ [test_batch0_pred.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp2_test_evaluation/test_batch0_pred.jpg)
* **Exp 4 (`YOLOv7-tiny Tuned`) 圖表**：[PR_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp4_test_evaluation/PR_curve.png) ｜ [F1_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp4_test_evaluation/F1_curve.png) ｜ [R_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp4_test_evaluation/R_curve.png) ｜ [confusion_matrix.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp4_test_evaluation/confusion_matrix.png) ｜ [test_batch0_pred.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/test/exp4_test_evaluation/test_batch0_pred.jpg)

| 測試集評估指標 (`test.py`) | 初版未調優 Exp 4 (`runs_origin`) | 調優後 **Exp 4 (`YOLOv7-tiny`, 100 ep)** | **Exp 2 (`YOLOv7 Base`, 50 ep)** | 核心發現與比較分析 |
| :--- | :---: | :---: | :---: | :--- |
| **全類別 Test mAP@0.5** | `0.146` | **`0.223`** (+52.7% vs 初版) | **`0.280`** 🏆 (+25.6% vs Exp 4) | 調優後 Exp 4 大幅躍升至 `0.223`；而大型骨幹 Exp 2 進一步推升至 **`0.280`** |
| **`traffic_light` AP@0.5** | `0.298` | `0.416` | **`0.508`** 🏆 | Exp 2 突破 0.50 大關，混淆矩陣分類正確率達 **58%**（Exp 4 為 48%） |
| **`Sign_follow` AP@0.5** | `0.144` | `0.352` | **`0.393`** 🏆 | 遵行標誌於調優後 Exp 4 成長 2.4 倍，Exp 2 再提升至 `0.393` |
| **`Sign_prohibit` AP@0.5** | `0.168` | `0.239` | **`0.385`** 🏆 | Exp 2 對圓形禁制標誌之特徵辨識力顯著領先（混淆矩陣檢出率 **53%** vs Exp 4 之 35%） |
| **`crosswalk` AP@0.5** | `0.003` | `0.048` | **`0.197`** 🏆 | Exp 2 的深層感受野使大範圍斑馬線 AP 較 Exp 4 高出 **4.1 倍** |
| **`Sign_limit` AP@0.5** | `0.260` | **`0.270`** 🏆 | `0.199` | 兩者皆能定位目標，但因紅圈特徵與樣本數 11:1 差距，常被歸入 `Sign_prohibit` |
| **`two_stage_left_turn_box` AP** | `0.000` | **`0.012`** 🏆 | `0.000` | **調參價值體現**：Exp 4 因放寬 `anchor_t=6.0` 且關閉旋轉剪切，成功產生微幅正響應 (`0.012`) |
| **最佳 F1-Score 與工作門檻** | `0.16` (@`0.052`) | `0.20` (@`0.239`) | **`0.25` (@`0.244`)** 🏆 | 兩組最佳信心門檻皆穩定收斂於 `~0.24`，對齊實務推論標準 (`0.25`) |
| **最大 Recall (`R_curve`)** | `0.51` | `0.63` | **`0.68`** 🏆 | Exp 2 在低門檻下最高可召回 **68%** 之測試集目標 |

---

## 三、實務推論逐框診斷對比 (`Step 7 & 8: detect.py @ conf >= 0.25, IoU >= 0.5`)

透過 [scripts/analyze_test_results.py](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/scripts/analyze_test_results.py) 比對測試集 16 張影像（共 84 個真值標註框）與模型實務推論結果（[detect_test_exp2/labels](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/detect/detect_test_exp2/labels) vs. [detect_test_images/labels](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/detect/detect_test_images/labels)）：

### 3.1 整體檢出與誤判統計表

| 實務推論指標 (`conf >= 0.25`) | 初版未調優 Exp 4 (`runs_origin`) | 調優後 **Exp 4 (`YOLOv7-tiny`, 100 ep)** | **Exp 2 (`YOLOv7 Base`, 50 ep)** | 差異比較 (`Exp 2` vs. 調優後 `Exp 4`) |
| :--- | :---: | :---: | :---: | :--- |
| **模型預測框總數 (Predictions)** | 51 個 | 65 個 | **56 個** | Exp 2 預測框更精簡（少 9 個框，卻抓對更多目標） |
| **✅ 成功辨識 (True Positives, TP)** | 16 個 (`19.0%` 召回) | **26 個 (`31.0%` 召回)** | **30 個 (`35.7%` 召回)** 🏆 | **Exp 2 多抓到 4 個目標**（比初版 Exp 4 成長 **+87.5%**） |
| **⚠️ 漏抓目標 (False Negatives, FN)** | 61 個 (`72.6%` 漏檢) | 55 個 (`65.5%` 漏檢) | **51 個 (`60.7%` 漏檢)** 🏆 | Exp 2 漏檢率最低（較初版降低 11.9 個百分點） |
| **❓ 誤判虛警 (False Positives, FP)** | 35 個 (`68.6%` 誤報) | 39 個 (`60.0%` 誤報) | **26 個 (`46.4%` 誤報)** 🏆 | **Exp 2 誤判虛警大減 13 個！整體實務 Precision 達 `53.6%`！** |
| **🔄 類別混淆 (Class Mismatch)** | 7 個 | **3 個** | **3 個** | 兩者皆僅 3 個同造型標誌（`Sign_limit`/`Sign_prohibit`/`Sign_follow`）混淆 |

---

### 3.2 各類別實務推論細部表現 (`conf >= 0.25, IoU >= 0.5`)

| 類別名稱 (Class) | 測試集真值數 (GT) | **Exp 4 (`YOLOv7-tiny Tuned`)**<br>TP / FN / FP (Recall / Precision) | **Exp 2 (`YOLOv7 Base`)**<br>TP / FN / FP (Recall / Precision) | 各類別勝負與關鍵原因分析 |
| :--- | :---: | :---: | :---: | :--- |
| **`traffic_light`** | 31 | 15 / 16 / 19 (`48.4%` R / `44.1%` P) | **17 / 14 / 12 (`54.8%` R / `58.6%` P)** 🏆 | **Exp 2 完勝**：多抓 2 個紅綠燈，且誤判從 19 個降至 12 個 |
| **`Sign_prohibit`** | 17 | 5 / 11 / 12 (`29.4%` R / `29.4%` P) | **10 / 6 / 8 (`58.8%` R / `55.6%` P)** 🏆 | **Exp 2 翻倍勝出**：抓到數量由 5 個倍增至 10 個，誤判亦減少 4 個 |
| **`Sign_follow`** | 13 | **5 / 8 / 7 (`38.5%` R / `41.7%` P)** 🏆 | 3 / 10 / 5 (`23.1%` R / `37.5%` P) | **Exp 4 勝出**：受惠於 `cls_pw=1.2` 與 `obj_pw=1.5` 強化少數類召回，多抓到 2 個遵行標誌 |
| **`crosswalk`** | 14 | **1 / 13 / 1 (`7.1%` R / `50.0%` P)** 🏆 | 0 / 14 / 1 (`0.0%` R / `0.0%` P) | **Exp 4 定位更準**：兩者於 `001057` 皆框出 1 條 `crosswalk (0.32)`，但 Exp 4 因 `anchor_t=6.0` 且無旋轉干擾，`IoU >= 0.5` 成功算入 TP；Exp 2 則因框略偏右 (`IoU < 0.5`) 被計為 FP |
| **`two_stage_left_turn_box`** | 7 | 0 / 7 / 0 (`0.0%` R) | 0 / 7 / 0 (`0.0%` R) | 在 `conf >= 0.25` 下兩者皆未達門檻（需更多訓練輪數與調參結合） |
| **`Sign_limit`** | 2 | 0 / 0 / 0（2 個皆定位成功但混淆為 `Sign_prohibit`） | 0 / 0 / 0（2 個皆定位成功但混淆為 `Sign_prohibit`） | 兩模型在 `000188` 與 `000229` 均精準框出遠處限速牌，惟受訓練集 89:8 樣本比影響歸入 `Sign_prohibit` |

---

## 四、16 張測試集影像逐張比對明細表（`Exp 2` vs. `Exp 4`）

| 測試集影像 ID | GT 數 | **Exp 2 (`YOLOv7 Base`, 50 ep)** 逐張檢出結果 ([查看圖片](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/detect/detect_test_exp2)) | **Exp 4 (`YOLOv7-tiny Tuned`, 100 ep)** 逐張檢出結果 ([查看圖片](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_exp2/detect/detect_test_images)) |
| :--- | :---: | :--- | :--- |
| **`KvNLWsBQYPE_000004_t00004`** | 12 | ✅ **抓到(7)**: `traffic_light`×4, `Sign_prohibit`×2, `Sign_follow`×1 ｜ ❓誤判(2) ｜ ⚠️漏抓(5) | ✅ **抓到(6)**: `traffic_light`×3, `Sign_follow`×2, `Sign_prohibit`×1 ｜ ❓誤判(5) ｜ ⚠️漏抓(6) |
| **`KvNLWsBQYPE_000188_t00188`** | 1 | 🔄 **混淆(1)**: 真值 `Sign_limit` ➔ 預測 `Sign_prohibit` (0.55) | 🔄 **混淆(1)**: 真值 `Sign_limit` ➔ 預測 `Sign_prohibit` (0.81) |
| **`KvNLWsBQYPE_000229_t00229`** | 1 | 🔄 **混淆(1)**: 真值 `Sign_limit` ➔ 預測 `Sign_prohibit` (0.35) | 🔄 **混淆(1)**: 真值 `Sign_limit` ➔ 預測 `Sign_prohibit` (0.64) |
| **`KvNLWsBQYPE_000956_t00956`** | 4 | ✅ 抓到(0) ｜ ❓誤判(1): `traffic_light` ｜ ⚠️漏抓(4) | ✅ 抓到(0) ｜ ❓誤判(2): `traffic_light`×2 ｜ ⚠️漏抓(4) |
| **`KvNLWsBQYPE_000967_t00967`** | 5 | ✅ **抓到(1)**: `traffic_light` (0.31) ｜ ❓誤判(1) ｜ ⚠️漏抓(4) | ✅ **抓到(1)**: `traffic_light` (0.35) ｜ ❓誤判(1) ｜ ⚠️漏抓(4) |
| **`KvNLWsBQYPE_001016_t01016`** | 4 | ✅ **抓到(1)**: `Sign_prohibit` (0.30) ｜ ❓誤判(1) ｜ ⚠️漏抓(3) | ✅ 抓到(0) ｜ ❓誤判(1): `traffic_light` ｜ ⚠️漏抓(4) |
| **`KvNLWsBQYPE_001057_t01057`** | 16 | ✅ **抓到(3)**: `Sign_prohibit`×2, `traffic_light`×1 ｜ ❓誤判(3, 含偏移 `crosswalk`) ｜ ⚠️漏抓(13) | ✅ **抓到(4)**: `Sign_prohibit`×1, `traffic_light`×2, **`crosswalk` (0.32)** ｜ ❓誤判(1) ｜ ⚠️漏抓(12) |
| **`KvNLWsBQYPE_001324_t01324`** | 10 | ✅ **抓到(3)**: `traffic_light`×2, `Sign_follow`×1 ｜ ❓誤判(1) ｜ ⚠️漏抓(7) | ✅ **抓到(4)**: `Sign_follow`×2, `traffic_light`×2 ｜ ❓誤判(4) ｜ ⚠️漏抓(6) |
| **`TTJkh6VwNgY_000463_t00463`** | 5 | ✅ **抓到(2)**: `traffic_light`×2 ｜ ❓誤判(2) ｜ ⚠️漏抓(3) | ✅ **抓到(1)**: `traffic_light`×1 ｜ ❓誤判(3) ｜ ⚠️漏抓(4) |
| **`TTJkh6VwNgY_000477_t00477`** | 0 | ❓ 誤判(1): `Sign_prohibit` (0.54) | ❓ 誤判(2): `Sign_prohibit` (0.76), `Sign_follow` (0.37) |
| **`ZKJjA2sFqTE_000028_t00028`** | 12 | ✅ **抓到(6)**: `Sign_prohibit`×3, `traffic_light`×2, `Sign_follow`×1 ｜ **❓誤判(0)** 🏆 ｜ ⚠️漏抓(6) | ✅ **抓到(3)**: `traffic_light`×1, `Sign_prohibit`×1, `Sign_follow`×1 ｜ ❓誤判(2) ｜ ⚠️漏抓(9) |
| **`ZKJjA2sFqTE_000052_t00052`** | 4 | ✅ **抓到(3)**: `Sign_prohibit`×1, `traffic_light`×2 ｜ ❓誤判(2) ｜ ⚠️漏抓(1) | ✅ **抓到(3)**: `traffic_light`×2, `Sign_prohibit`×1 ｜ ❓誤判(1) ｜ ⚠️漏抓(1) |
| **`ZKJjA2sFqTE_000190_t00190`** | 0 | ❓ 誤判(4) | ❓ 誤判(5) |
| **`ZKJjA2sFqTE_001261_t01261`** | 2 | ✅ 抓到(0) ｜ ❓誤判(2) ｜ ⚠️漏抓(2) | ✅ 抓到(0) ｜ ❓誤判(4) ｜ ⚠️漏抓(2) |
| **`ZKJjA2sFqTE_001738_t01738`** | 4 | ✅ **抓到(2)**: `Sign_prohibit`×1, `traffic_light`×1 ｜ 🔄混淆(1) ｜ ❓誤判(1) ｜ ⚠️漏抓(1) | ✅ **抓到(1)**: `Sign_prohibit`×1 ｜ 🔄混淆(1) ｜ ❓誤判(3) ｜ ⚠️漏抓(2) |
| **`ZKJjA2sFqTE_001908_t01908`** | 4 | ✅ **抓到(2)**: `traffic_light`×2 ｜ ❓誤判(2) ｜ ⚠️漏抓(2) | ✅ **抓到(3)**: `traffic_light`×3 ｜ ❓誤判(2) ｜ ⚠️漏抓(1) |

---

## 五、為什麼下一步要將「Exp 4 改為 `YOLOv7 Base` + 調優後超參數 + 100 Epochs」？

從上述對比可以清楚看見**「互補現象」**：
1. **`Exp 2 (YOLOv7 Base, 50 ep, 預設超參數)` 的強項**：網路容量大、抗背景雜訊能力強（誤判 FP 比 Tiny 少 13 個，且 `traffic_light` 與 `Sign_prohibit` 檢出率極高）；但因為**只訓練了 50 輪**，且使用的是**預設超參數（含有 `degrees`/`shear`/`mixup` 且 `anchor_t=4.0`）**，導致它在 `KvNLWsBQYPE_001057` 框出的 `crosswalk` 邊界框不夠貼合（`IoU < 0.5`），且 `two_stage_left_turn_box` 完全無響應。
2. **`Exp 4 (YOLOv7-tiny, 100 ep, 調優超參數)` 的強項**：我們在 [configs/hyp.scratch.custom.yaml](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml) 設定的 **`anchor_t=6.0`、關閉旋轉剪切與 Mixup、提高 `obj_pw=1.5` / `cls_pw=1.2`**，成功讓輕量模型也能以 `IoU >= 0.5` 精準框出 `crosswalk` 並喚醒 `two_stage_left_turn_box (AP=0.012)`，同時在 `Sign_follow` 拿到更高的實務召回率；但受限於 `YOLOv7-tiny` 僅 6.0M 參數，容易把複雜街景招牌誤判為號誌（FP=39）。

👉 **強強聯手（Next Step）**：將 **Exp 4 升級為 `YOLOv7 Base` (`cfg/training/yolov7.yaml` + `yolov7.pt`) + 100 Epochs + 調優後的 [configs/hyp.scratch.custom.yaml](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml)**，即可結合「大型骨幹的低誤判率與大感受野」以及「100 輪充分收斂 + 扁平標線友善超參數」的雙重優勢！
