# NCHU Research Method - Taiwan Traffic Facility Detection Dataset & Pipeline

[![Course](https://img.shields.io/badge/NCHU-Research%20Methodology-blue.svg)](https://www.nchu.edu.tw/)
[![Model Architecture](https://img.shields.io/badge/Framework-YOLOv7-green.svg)](https://github.com/contemplator/nchu-research-method-traffic-violation-model)
[![License](https://img.shields.io/badge/License-Academic%20Use%20Only-orange.svg)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contemplator/nchu-research-method-traffic-violation-dataset/blob/feat/458-images/notebooks/NCHU_Research_YOLOv7.ipynb)
[![Notion Guide](https://img.shields.io/badge/Notion-YOLO7%20操作指南-000000.svg?logo=notion&logoColor=white)](https://app.notion.com/p/YOLO7-3fafe2e0792381c680b2c415a192ccd3?source=copy_link)

本專案為國立中興大學「**研究方法 (Research Methodology)**」期末專題（Part 2: Detection with YOLO）之資料工程、前處理工具與消融實驗管理儲存庫。

針對台灣市區道路情境中常見之**交通號誌、路面標線與交通標誌等關鍵交通設施偵測（Traffic Facility Detection）**，本分支（`feat/458-images`）建置客製化 **6 類別、458 張影像**之標註資料集，並搭配 [nchu-research-method-traffic-violation-model](https://github.com/contemplator/nchu-research-method-traffic-violation-model) 於 Google Colab 進行系統化深度學習消融實驗（`Exp 1 ~ Exp 10` Append-Only Ablation Study）與 4 維度誤差診斷分析。

---

## 📌 專案目錄結構 (Repository Structure)

```text
nchu-research-method-traffic-violation-dataset/
├── README.md                                  # [本文件] 專案總覽、6 類別資料集與消融實驗說明
├── configs/
│   ├── traffic_facility.yaml                  # YOLOv7 資料集定義設定檔 (路徑與 6 類別)
│   ├── traffic_facility_clahe.yaml            # LAB-CLAHE 前處理版資料集設定檔 (供 Exp 10 使用)
│   ├── yolov7-tiny-custom.yaml                # 自訂模型架構檔 (nc=6)
│   ├── hyp.scratch.custom.yaml.original       # 初版領域調優超參數設定檔 (供 Exp 4 歷史對照)
│   ├── hyp.scratch.custom.yaml                # 領域調優 v1 超參數設定檔 (供 Exp 5, Exp 6 使用)
│   ├── hyp.traffic_highres.yaml               # 960 高解析度版超參數設定檔 (供 Exp 8 使用)
│   └── hyp.traffic_v2.yaml                    # 領域進階版 v2 超參數設定檔 (供 Exp 9, Exp 10 使用)
├── docs/
│   └── NEXT_STAGE_EXPERIMENT_PLAN.md          # 458 張完整資料集與 Exp 1~10 消融實驗規劃書
├── scripts/
│   ├── extract_frames.py                      # 行車記錄器影片定頻抽幀工具
│   ├── verify_annotations.py                  # 標註格式合規性檢查、類別分佈與負樣本防呆工具
│   ├── split_dataset.py                       # 7:2:1 (Train/Val/Test) 隨機切分與自動打包工具
│   ├── preprocess_clahe.py                    # LAB 色彩空間 L 通道 CLAHE 對比度增強工具
│   └── analyze_test_results.py                # 總對照表、4 象限收斂圖、4 維度誤差與推論對比矩陣工具
├── notebooks/
│   └── NCHU_Research_YOLOv7.ipynb             # Google Colab 雲端訓練、評估與消融分析標準筆記本
├── data/
│   ├── annotated/                             # 已標註之有效圖片 (.jpg)、原始標註檔 (.json) 與 classes.txt
│   ├── labels/                                # 匯出之標準 YOLO 標籤檔 (.txt)
│   ├── dataset/                               # 切分後之標準 YOLO 格式 (train/val/test)
│   ├── dataset_clahe/                         # CLAHE 前處理後之影像副本 (供 Exp 10 使用)
│   └── dataset.zip                            # 一鍵自動打包產生之雲端訓練壓縮檔
├── sample_videos/                             # 測試用台灣市區行車記錄器短片
└── results/                                   # 消融實驗輸出圖表、混淆矩陣與最佳權重備份
```

---

## 🏷️ 標註類別定義 (Target Classes)

本資料集定義台灣道路行車情境最常見之 **6 大交通設施類別**，統一採用 YOLO 正規化標註格式（對應 `configs/traffic_facility.yaml` 與 `data/annotated/classes.txt`）：

| 類別 ID (`class_id`) | 標籤名稱 (`name`) | 中文類別 | 標註框數量 | 標註要點與邊界框規範 |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `traffic_light` | 交通號誌（紅綠燈） | 1,212 | 道路上方或路側之行車管制號誌、行人號誌等燈箱輪廓。 |
| **1** | `crosswalk` | 行人穿越道（斑馬線） | 442 | 路口路面繪設之枕木紋／斑馬紋行人穿越道標線區域。 |
| **2** | `two_stage_left_turn_box` | 機車兩段式左轉待轉區 | 76 | 路口路面繪設之白色矩形機車待轉格標線。 |
| **3** | `Sign_follow` | 遵行標誌 | 397 | 圓形藍底白圖之指示／遵行交通標誌（如單行道、靠右行駛、機慢車兩段左轉等）。 |
| **4** | `Sign_prohibit` | 禁制標誌 | 488 | 圓形紅框白底或紅底之禁止類交通標誌（如禁止左轉、禁止停車、禁止進入等）。 |
| **5** | `Sign_limit` | 速限／限制標誌 | 37 | 圓形紅框白底數字之最高／最低速限等限制類交通標誌。 |

* **標籤格式**：每一行代表一個目標，數值經圖片長寬正規化至 `[0, 1]` 區間：
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
* **負樣本處理**：不含任何目標之道路影像保留空白 `.txt` 標註檔，以協助模型抑制背景假陽性（False Positives）。

---

## 📊 資料集規模與劃分 (Dataset Scale & Split)

* **完整影像總數**：**458 張**台灣真實道路行車記錄器影像（457 張含目標標註影像 + 1 張無目標負樣本）。
* **總標註框數量**：**2,652 個**標註目標（Instances），涵蓋常見高頻類別（`traffic_light`）與長尾少數類別（`two_stage_left_turn_box`、`Sign_limit`）。
* **切分比例（固定 `seed=42`，`7 : 2 : 1`）**：
  * **Train (訓練集)**：320 張（70%）
  * **Val (驗證集)**：91 張（20%，用於訓練驗證與 Validation-First 最佳 F1 信心門檻搜尋）
  * **Test (測試集)**：47 張（10%，獨立保留盲測評估集，嚴格防止資料洩漏）

---

## 🛠️ 本地資料處理工作流程 (Data Preprocessing Workflow)

> 📖 **完整圖文手冊**：詳細的 X-AnyLabeling 介面設定、標註標準與常見問題避坑，請同步參閱 [Notion - YOLO7 操作指南](https://app.notion.com/p/YOLO7-3fafe2e0792381c680b2c415a192ccd3?source=copy_link)。

### 1. 影片抽幀收集
```bash
python scripts/extract_frames.py --video sample_videos/dashcam.mp4 --output data/raw/ --interval 2
```

### 2. 本地影像標註與匯出
推薦使用 **[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)**：
- **開啟圖片目錄**：`data/annotated/`。
- **標註規範**：繪製矩形框緊貼號誌、標誌或標線輪廓，依序標註 6 大交通設施類別。
- **匯出 YOLO HBB 標籤**：
  - 點擊 **「導出」$\rightarrow$「YOLO HBB」**。
  - **選取類別定義檔**：選擇 `data/annotated/classes.txt`（作為文字標籤轉譯為 `0~5` 數字代號之對照字典）。
  - **導出路徑**：選擇 **`data/labels/`**（避免覆蓋原圖目錄）。
  - **「同時保存圖片？」**：**不用勾選**（原圖已在 `annotated/`，無需重複複製）。

### 3. 標註資料防呆檢驗
執行檢查腳本，自動跨目錄比對圖片與標籤，確認無缺漏、無孤兒檔、數值介於 `0.0 ~ 1.0`，並輸出各類別分佈與負樣本統計：
```bash
# 檢查原始標註目錄
python scripts/verify_annotations.py

# 或於切分後檢查 train / val / test 全 split
python scripts/verify_annotations.py --dataset_dir data/dataset
```
- **核心功能**：自動抓取 `data/annotated/`（圖片）與 `data/labels/`（標籤）進行成對校驗，或針對 `data/dataset/` 進行全 split 驗證。
- **檢查重點**：確認輸出「0 錯誤」，且終端機顯示 6 類別數量統計分佈與負樣本數。

### 4. 一鍵資料集切分與自動打包
依 7:2:1（`seed=42`）自動隨機切分至 `data/dataset/{train,val,test}`，並**自動生成 `data/dataset.zip`**：
```bash
python scripts/split_dataset.py
```
- **全自動固定邏輯流程**：
  1. 自動讀取 `data/annotated/` 與 `data/labels/` 並成對合併。
  2. 依 7:2:1 比例隨機切分至 `data/dataset/train/`、`val/`、`test/`（320 / 91 / 47）。
  3. 自動使用純 Python 壓縮打包產出 **`data/dataset.zip`**（跨平台標準，內含標準 `dataset/` 頂層結構，免除手動 zip 操作）。

### 5. LAB-CLAHE 局部對比度增強（選用，供 Exp 10）
針對逆光、樹蔭與夜間低對比度場景，可於 LAB 色彩空間之 `L` 通道套用 `CLAHE (clipLimit=2.0, tileGridSize=(8,8))`，輸出獨立副本至 `data/dataset_clahe/`（不覆蓋原圖，於 Colab 流程中亦會自動執行）：
```bash
python scripts/preprocess_clahe.py
```

---

## 🔬 消融實驗規劃 (Ablation Study Matrix)

本分支採用 **Append-Only 消融實驗架構（`Exp 1 ~ Exp 10`）**，完整保留歷史基準設定（`Exp 1 ~ Exp 6`），並往後累加高解析度、領域超參數 v2 與 LAB-CLAHE 前處理等進階實驗（`Exp 7 ~ Exp 10`）。詳細設計原則請參閱 [docs/NEXT_STAGE_EXPERIMENT_PLAN.md](docs/NEXT_STAGE_EXPERIMENT_PLAN.md)。

### 1. 保留基準組 (`Exp 1 ~ Exp 6`)

| 組別 | 模型架構 | 解析度 | Batch | Epochs | 超參數設定檔 | 實驗定位與探討目標 |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **Exp 1** | YOLOv7-tiny | 640 | 16 | 50 | `data/hyp.scratch.tiny.yaml` | 輕量模型 50 輪基準線 (Baseline) |
| **Exp 2** | YOLOv7 Base | 640 | 8 | 50 | `data/hyp.scratch.p5.yaml` | 大骨幹模型 50 輪尺度對照 |
| **Exp 3** | YOLOv7-tiny | 640 | 16 | 100 | `data/hyp.scratch.tiny.yaml` | 輕量模型 100 輪收斂基準 |
| **Exp 4** | YOLOv7-tiny | 640 | 16 | 100 | `configs/hyp.scratch.custom.yaml.original` | 輕量模型初版領域調優（歷史負向調參對照） |
| **Exp 5** | YOLOv7-tiny | 640 | 16 | 100 | `configs/hyp.scratch.custom.yaml` | 輕量模型領域調優 v1（對比 Exp 3 / Exp 4 並量測邊緣 FPS） |
| **Exp 6** | YOLOv7 Base | 640 | 8 | 100 | `configs/hyp.scratch.custom.yaml` | 大骨幹 + 100 輪 + 領域調優 v1 |

### 2. 進階對照組 (`Exp 7 ~ Exp 10`)

| 組別 | 模型架構 | 解析度 | Batch | Epochs | 超參數 / 前處理 | 核心對照關係與驗證假說 |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **Exp 7** | YOLOv7 Base | 640 | 8 | 100 | `data/hyp.scratch.p5.yaml` | 補齊 Base 100 輪預設參數對照（對比 Exp 2 驗輪數、對比 Exp 6 驗 v1 調參） |
| **Exp 8** | YOLOv7 Base | **960** | 8 | 100 | `configs/hyp.traffic_highres.yaml` | 對比 Exp 6，單一解析度變因（`640 → 960`，微調 `lr0=0.005`）驗證遠距小標誌偵測效益 |
| **Exp 9** | YOLOv7 Base | **960** | 8 | 100 | `configs/hyp.traffic_v2.yaml` | 對比 Exp 8，加入 Focal Loss (`fl_gamma=1.0`)、關閉水平翻轉 (`fliplr=0.0`)、降 `mosaic=0.5`、升 `box=0.075` 改善長尾混淆與標線定位 |
| **Exp 10** | YOLOv7 Base | **960** | 8 | 100 | `configs/hyp.traffic_v2.yaml` + **LAB-CLAHE** | 對比 Exp 9，套用 LAB-CLAHE 局部對比度增強（`configs/traffic_facility_clahe.yaml`），驗證逆光／陰影／夜間場景召回提升 |

* **評估協議**：所有實驗皆採 **Validation-First Thresholding**（先於 Val 集 91 張透過 F1 Curve 決定最佳信心門檻 `best_conf`，再統一套用至 Test 集 47 張進行盲測），並自動產出 Master Benchmark 總表、4 象限收斂曲線、4 維度誤差分類統計（`FP`、`FN`、`Misclassification`、`Low IoU`）與同圖並排推論對比矩陣。

---

## 🚀 Google Colab 雲端訓練操作簡介

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contemplator/nchu-research-method-traffic-violation-dataset/blob/feat/458-images/notebooks/NCHU_Research_YOLOv7.ipynb)

1. 將本地一鍵生成之 **`data/dataset.zip`** 上傳至個人 Google Drive 的 `NCHU_YOLO_Project_Traffic_Facility_6_classes/` 目錄中（即 `NCHU_YOLO_Project_Traffic_Facility_6_classes/dataset.zip`）。
2. 點擊上方按鈕直接於 Google Colab 開啟 [notebooks/NCHU_Research_YOLOv7.ipynb](https://colab.research.google.com/github/contemplator/nchu-research-method-traffic-violation-dataset/blob/feat/458-images/notebooks/NCHU_Research_YOLOv7.ipynb)。
3. 依序執行儲存格：
   * 掛載 Google Drive
   * Clone 模型倉庫 [nchu-research-method-traffic-violation-model](https://github.com/contemplator/nchu-research-method-traffic-violation-model) 與 `feat/458-images` 分支
   * 自動解壓縮 `dataset.zip`、執行標註品質驗證與 LAB-CLAHE 前處理，並啟動 `Exp 1 ~ Exp 10` 結構化消融訓練迴圈
   * 執行 Validation-First 門檻搜尋、Test 盲測評估、4 維度誤差診斷與推論對比矩陣，並一鍵打包回傳成果 (`results_summary_458_exp1_to_10.zip`)

---

## 🔗 相關專案連結

* 📋 實驗規劃書：[docs/NEXT_STAGE_EXPERIMENT_PLAN.md](docs/NEXT_STAGE_EXPERIMENT_PLAN.md)
* 📖 團隊操作指南：[Notion - YOLO7 操作指南 (完整圖文手冊)](https://app.notion.com/p/YOLO7-3fafe2e0792381c680b2c415a192ccd3?source=copy_link)
* 模型庫：[contemplator/nchu-research-method-traffic-violation-model](https://github.com/contemplator/nchu-research-method-traffic-violation-model)
* 上游官方庫：[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
