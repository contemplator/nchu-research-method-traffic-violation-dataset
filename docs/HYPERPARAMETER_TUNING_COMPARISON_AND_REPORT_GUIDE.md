# 📊 YOLOv7-tiny 超參數調整前後差異比較與報告撰寫指南 (`runs_origin` vs. `runs_hyp`)

> **建立日期**：2026-11-16  
> **對照實驗**：Exp 4 (`exp4_yolov7_tiny_tuned`, 100 Epochs, `YOLOv7-tiny`)  
> **調參設定檔對照**：[hyp.scratch.custom.yaml.original](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml.original)（調整前 `runs_origin`） vs. [hyp.scratch.custom.yaml](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/configs/hyp.scratch.custom.yaml)（調整後 `runs_hyp`）  
> **資料集規模**：152 張台灣街景影像（Train: 106 張 / Val: 30 張 / Test: 16 張，共 84 個測試集真值目標），6 大交通設施與標誌類別

---

## 一、超參數調整明細與物理意義

在初版實驗中，**Exp 4 調整前（`runs_origin`, Val mAP@0.5 = `0.161`）** 反而低於使用官方預設超參數的 **Exp 3（Val mAP@0.5 = `0.206`）**，屬於**負向調參**（主要原因為模型過度保守導致大量漏檢，Val Recall 僅 `0.150`）。針對台灣街景「小樣本」與「極扁平水平地面標線（斑馬線、待轉區）」的特性，調整後（`runs_hyp`）修改了以下核心超參數：

| 參數名稱 | 調整前 (`runs_origin`) | 調整後 (`runs_hyp`) | 調整原因與物理意義 |
| :--- | :---: | :---: | :--- |
| **`lr0`** | `0.005` | **`0.01`** | 106 張訓練集在 `batch=16` 下每 Epoch 僅 7 個迭代步，100 Epochs 共約 700 次更新；原學習率過小導致欠擬合，恢復 `0.01` 以加速收斂。 |
| **`degrees` / `shear` / `perspective`** | `5.0` / `2.0` / `0.0005` | **`0.0` / `0.0` / `0.0`** | 斑馬線 (`crosswalk`) 與待轉區 (`two_stage_left_turn_box`) 為極扁平水平框，旋轉與剪切會使水平外接框膨脹並混入大量柏油路面背景雜訊。 |
| **`mixup` / `paste_in`** | `0.1` / `0.05` | **`0.0` / `0.0`** | 僅保留 `mosaic: 1.0` 維持小目標尺度多樣性，關閉 Mixup 與 Paste-in 避免合成疊加訊號過於複雜而干擾輕量模型收斂。 |
| **`anchor_t`** | `4.0` | **`6.0`** | 放寬 Anchor 長寬比匹配門檻，讓極端長寬比的地面標線能匹配到更多正樣本預測框。 |
| **`obj_pw` / `cls_pw`** | `1.0` / `1.0` | **`1.5` / `1.2`** | 提高前景 Objectness 與少數類別 Classification 的正樣本損失權重，改善模型過度保守造成的大量漏檢（FN）。 |
| **`translate`** | `0.15` | `0.1` | 適度縮減平移幅度，避免邊界附近的小目標標誌在增強時被過度裁切。 |

---

## 二、調整前後量化結果全面比較

### 1. 驗證集（Val Split, 30 張影像）訓練收斂對比
資料來源：[runs_origin/train/exp4_yolov7_tiny_tuned/results.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/train/exp4_yolov7_tiny_tuned/results.png) vs. [runs_hyp/train/exp4_yolov7_tiny_tuned/results.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_hyp/train/exp4_yolov7_tiny_tuned/results.png)，以及 [調參前混淆矩陣](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/train/exp4_yolov7_tiny_tuned/confusion_matrix.png) vs. [調參後混淆矩陣](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_hyp/train/exp4_yolov7_tiny_tuned/confusion_matrix.png)。

| 驗證集指標 (Val Split) | Exp 3（預設 hyp, 100 ep） | 調整前 Exp 4 (`runs_origin`) | 調整後 Exp 4 (`runs_hyp`) | 調參前後變化（Exp 4） |
| :--- | :---: | :---: | :---: | :--- |
| **mAP@0.5** | `0.206` | `0.161` | **`~0.225`** | **+39.8%**（正式超越不調參的 Exp 3！） |
| **mAP@0.5:0.95** | `0.0925` | `0.0629` | **`~0.102`** | **+62.2%**（邊界框定位精度大幅提升） |
| **Precision** | `0.680` | **`0.756`** | `~0.65` | 下降約 10 個百分點（更積極預測前景的 Precision–Recall 權衡） |
| **Recall** | `0.217` | `0.150` | **`~0.228`** | **+52.0%**（大幅改善原本過度保守的漏檢） |
| **`crosswalk` 檢出率** | `8%` (`0.08`) | `3%` (`0.03`) | **`25%` (`0.25`)** | **成長 8.3 倍**（漏檢率從 97% 降至 75%） |
| **`Sign_follow` 檢出率** | `37%` (`0.37`) | `14%` (`0.14`) | **`37%` (`0.37`)** | **成長 2.6 倍**（漏檢率從 86% 降至 63%） |
| **`Sign_prohibit` 檢出率** | `44%` (`0.44`) | `38%` (`0.38`) | **`44%` (`0.44`)** | 穩定提升 +6 個百分點 |
| **`traffic_light` 檢出率** | `44%` (`0.44`) | `34%` (`0.34`) | **`42%` (`0.42`)** | 穩定提升 +8 個百分點 |

---

### 2. 獨立測試集（Test Split, 16 張影像 / 84 個真值目標）評估曲線對比 (`test.py`)
資料來源：[調參前 PR_curve](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/test/exp4_test_evaluation/PR_curve.png) vs. [調參後 PR_curve](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_hyp/test/exp4_test_evaluation/PR_curve.png)、[調參前 F1_curve](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/test/exp4_test_evaluation/F1_curve.png) vs. [調參後 F1_curve](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_hyp/test/exp4_test_evaluation/F1_curve.png)。

| 測試集指標 (`test.py`) | 調整前 (`runs_origin`) | 調整後 (`runs_hyp`) | 變化與分析 |
| :--- | :---: | :---: | :--- |
| **全類別 Test mAP@0.5** | `0.146` | **`0.196`** | **相對提升 +34.2%** |
| **`traffic_light` AP@0.5** | `0.298` | **`0.407`** | **+36.6%**（主力類別表現最穩健） |
| **`Sign_follow` AP@0.5** | `0.144` | **`0.309`** | **+114.6%（翻倍成長）** |
| **`Sign_prohibit` AP@0.5** | `0.168` | **`0.245`** | **+45.8%** |
| **`crosswalk` AP@0.5** | `0.003` | **`0.070`** | **成長 23 倍**（於 [test_batch0_pred.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_hyp/test/exp4_test_evaluation/test_batch0_pred.jpg) 已可見橘色斑馬線預測框） |
| **`two_stage_left_turn_box` AP@0.5** | `0.000` | **`0.002`** | 由完全無響應突破至微幅檢出，但仍受限於樣本稀缺 |
| **`Sign_limit` AP@0.5** | **`0.260`** | `0.145` | **類別混淆副作用**：位置皆抓對，但因調高 `obj_pw`/`cls_pw` 而更強烈歸入外觀相似且樣本多 11 倍的 `Sign_prohibit` |
| **最佳 F1-Score 與信心門檻** | `0.16` (at `conf=0.052`) | **`0.21`** (at **`conf=0.242`**) | 最佳工作門檻由異常偏低的 `0.052` 校正回與實務預設 (`0.25`) 一致的 `0.242` |
| **最大 Recall (`R_curve`)** | `0.51` | **`0.61`** | 測試集極限檢出能力提升 **+10 個百分點** |

---

### 3. 實務推論比對（`detect.py`, `conf >= 0.25`, `IoU >= 0.5`）
由 [scripts/analyze_test_results.py](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/scripts/analyze_test_results.py) 逐框比對測試集 16 張影像（84 個真值目標）：

| 實務推論統計（`conf >= 0.25`） | 調整前 (`runs_origin`) | 調整後 (`runs_hyp` 純淨推論) | 改善與副作用說明 |
| :--- | :---: | :---: | :--- |
| **模型預測框總數 (Predictions)** | 51 個 | **70 個** | 模型不再過度保守，能於標準門檻輸出足夠前景預測 |
| **✅ 成功抓到 (True Positives, TP)** | 16 個 (`19.0%` 召回) | **24 個 (`28.6%` 召回)** | **成功檢出目標數增加 +50%（多抓到 8 個真實物件）** |
| **⚠️ 漏抓目標 (False Negatives, FN)** | 61 個 (`72.6%` 漏檢) | **55 個 (`65.5%` 漏檢)** | 漏檢率下降 7.1 個百分點 |
| **❓ 誤判虛警 (False Positives, FP)** | **35 個** (`68.6%` 誤報) | 46 個 (`65.7%` 誤報) | 預測變積極使絕對誤判數增加 11 個（Precision–Recall 權衡） |
| **🔄 類別混淆 (Class Mismatch)** | 7 個 | **5 個** | 定位正確但分類錯誤的案例減少 2 個 |
| **`Sign_follow` 實務檢出 (TP)** | `0 個` (`0.0%` Recall) | **`4 個` (`30.8%` Recall)** | **從完全漏檢突破至 30.8% 召回率** |
| **`traffic_light` 實務檢出 (TP)** | `11 個` (`35.5%` R / `37.9%` P) | **`15 個` (`48.4%` R / `41.7%` P)** | **Recall (+12.9%) 與 Precision (+3.8%) 雙雙提升** |

---

## 三、誤差與副作用分析（Error Analysis & Trade-offs）

寫報告時，主動分析**「哪些地方變好、哪些地方變差、為什麼」**能大幅提升研究可信度：

1. **Precision–Recall 權衡（誤判虛警 FP 增加）**：
   * 調升 `obj_pw=1.5` 後，模型對前景目標更敏感，成功救回 8 個原本漏抓的真實物件（TP 16 $\rightarrow$ 24），但同時也讓 `conf >= 0.25` 下的背景誤判數由 35 個增至 46 個（例如兩張無目標空背景圖 `TTJkh6VwNgY_000477` 與 `ZKJjA2sFqTE_000190` 之誤判數由共 4 個增至共 6 個），Val Precision 亦由 `0.756` 降至約 `0.65`。
2. **長尾類別細分類混淆（`Sign_limit` AP 下降）**：
   * 在 `KvNLWsBQYPE_000188` 與 `KvNLWsBQYPE_000229` 兩張僅含 `Sign_limit` 的測試影像中，模型**已精準定位目標位置 (`IoU >= 0.5`)，並無漏檢**；但因訓練集中 `Sign_prohibit`（89 個）與 `Sign_limit`（8 個）存在 **11:1 的樣本不平衡**，且兩者同為「紅圈白底圓牌」，提高正樣本權重後使模型更強烈將其分類為 `Sign_prohibit`（信心度由 `0.75 / 0.46` 升至 `0.84 / 0.61`），導致 `Sign_limit` AP 由 `0.260` 降至 `0.145`。

> 💡 **技術細節備忘**：`runs_hyp/train/exp4_yolov7_tiny_tuned/results.png` 因當時使用 `--exist-ok` 重跑，X 軸呈現 `0~200`（`0~99` 為調整前、`100~199` 為調整後）；而 `runs_hyp/detect/detect_test_images/labels/*.txt` 亦因附加模式保留了調整前 51 行與調整後 70 行（上表已濾除重複前綴）。後續已在 [NCHU_Research_YOLOv7.ipynb](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/notebooks/NCHU_Research_YOLOv7.ipynb) 的 Step 5、6、7、9 加入 `!rm -rf` / `!rm -f` 防呆指令，避免重跑時再次發生檔案疊加。

---

## 四、期末報告撰寫範本（可直接改寫使用）

### 4.1 實驗章節論述架構：「問題診斷 $\rightarrow$ 調校策略 $\rightarrow$ 結果驗證」

> **【問題診斷範本】**  
> 在初步消融實驗中，初次自訂超參數之 `Exp 4 (runs_origin)` 於驗證集與測試集之 `mAP@0.5` 分別僅達 `0.161` 與 `0.146`，低於採用預設超參數之 `Exp 3 (0.206)`，呈現負向調參現象。經剖析訓練曲線與混淆矩陣，歸納出三大主因：(1) 在僅 106 張訓練影像的小樣本條件下，過低之學習率（`lr0=0.005`）搭配 `mixup=0.1` 導致輕量模型 `YOLOv7-tiny` 收斂不足；(2) 開啟旋轉（`degrees=5.0`）與剪切（`shear=2.0`）破壞了台灣道路斑馬線與待轉區等極扁平水平標線之邊界框幾何結構，且 `anchor_t=4.0` 過嚴使標線難以匹配正樣本（測試集 `crosswalk` AP 僅 `0.003`）；(3) 預設正負樣本權重使模型過度保守，測試集最佳 F1 信心門檻異常偏低（`0.052`），於一般推論門檻（`0.25`）下產生高達 `72.6%` 之漏檢率。

> **【調校策略與驗證範本】**  
> 針對上述瓶頸，本研究提出領域知識導向之超參數修正：（一）將 `lr0` 恢復為 `0.01` 並關閉 `mixup` 與 `paste_in`，僅保留 `mosaic=1.0` 以維持小目標尺度多樣性；（二）關閉破壞水平標線幾何之旋轉與剪切（`degrees=0.0, shear=0.0, perspective=0.0`），並將 `anchor_t` 放寬至 `6.0`；（三）調升正樣本損失權重（`obj_pw=1.5, cls_pw=1.2`）以校正保守推論。實驗結果證實，調參後之 `Exp 4 (runs_hyp)` 在驗證集 `mAP@0.5` 由 `0.161` 躍升至 `0.225`（+39.8%，成功超越 `Exp 3` 之 `0.206`），測試集 `mAP@0.5` 由 `0.146` 提升至 `0.196`（+34.2%）；於標準推論門檻（`conf=0.25`）下，真實目標檢出數量（TP）成長 **50%**（16 $\rightarrow$ 24 個），且最佳 F1 信心門檻成功校正回實務標準的 `0.242`。

* **建議報告並排圖表**：
  1. [runs_origin PR_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/test/exp4_test_evaluation/PR_curve.png) vs. [runs_hyp PR_curve.png](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_hyp/test/exp4_test_evaluation/PR_curve.png)
  2. [runs_origin test_batch0_pred.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_origin/test/exp4_test_evaluation/test_batch0_pred.jpg) vs. [runs_hyp test_batch0_pred.jpg](file:///Users/leolin/NCHU/nchu-research-method-traffic-violation-dataset/results/runs_hyp/test/exp4_test_evaluation/test_batch0_pred.jpg)（可明顯指出 `crosswalk` 與 `Sign_follow` 的檢出突破）

---

### 4.2 期末報告結論（Conclusion & Future Work 三層次範本）

1. **領域先驗超參數調校突破輕量模型瓶頸**：  
   本研究證實於小樣本台灣街景資料集（152 張）中，通用型幾何形變增強（旋轉、剪切）與 Mixup 會破壞極端長寬比道路標線之特徵並阻礙收斂。透過關閉破壞性形變、放寬 Anchor 匹配門檻（`anchor_t=6.0`）並強化前景與少數類正樣本權重（`obj_pw=1.5, cls_pw=1.2`），成功使 `YOLOv7-tiny` 測試集 `mAP@0.5` 提升 **34.2%**（`0.146` $\rightarrow$ `0.196`），實務檢出目標數成長 **50%**，並將最佳信心門檻由 `0.052` 校正至 `0.242`。
2. **模型骨幹容量與感受野之權衡**：  
   綜合四組消融實驗，經超參數優化之 `YOLOv7-tiny`（12.3 MB，100 Epochs，Val mAP@0.5 `~0.225`）展現極佳輕量化潛力；然面對遠距微小號誌與大面積透視變形斑馬線時，標準版 `YOLOv7 Base`（74.8 MB）僅訓練 50 Epochs 即達 `0.244` 最高指標，顯示更深的 E-ELAN 骨幹與較大感受野在抵抗複雜街景背景雜訊（如招牌、建築物）上仍具顯著優勢。
3. **長尾分佈限制與未來展望**：  
   誤差分析顯示模型已能準確定位稀缺類別（如 `Sign_limit` 位置重疊率 `IoU > 0.5`），但受限於 `Sign_prohibit` 與 `Sign_limit` 高達 11:1 的樣本不平衡與紅圈相似特徵，調升正樣本權重後亦伴隨背景誤判（FP）略增與細分類混淆之權衡。未來工作建議：(1) 針對 `Sign_limit` 與 `two_stage_left_turn_box` 進行針對性樣本擴充或啟用 Focal Loss（`fl_gamma > 0`）；(2) 將本次驗證有效之領域超參數配置遷移至 `YOLOv7 Base` 並訓練至 100 Epochs，以兼顧高召回率與高精度定位。
