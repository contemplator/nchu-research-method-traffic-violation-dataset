# NCHU Research Method - Taiwan Traffic Violation Dataset & Pipeline

[![Course](https://img.shields.io/badge/NCHU-Research%20Methodology-blue.svg)](https://www.nchu.edu.tw/)
[![Model Architecture](https://img.shields.io/badge/Framework-YOLOv7-green.svg)](https://github.com/contemplator/nchu-research-method-traffic-violation-model)
[![License](https://img.shields.io/badge/License-Academic%20Use%20Only-orange.svg)](#)

本專案為國立中興大學「**研究方法 (Research Methodology)**」期末專題（Part 2: Detection with YOLO）之資料工程、前處理工具與消融實驗管理儲存庫。

針對台灣市區常見之交通違規停車痛點，本專案建置客製化標註資料集，並搭配 [nchu-research-method-traffic-violation-model](https://github.com/contemplator/nchu-research-method-traffic-violation-model) 於 Google Colab 進行深度學習消融實驗（Ablation Study）與行車記錄器動態影片推論。

---

## 📌 專案目錄結構 (Repository Structure)

```text
nchu-research-method-traffic-violation-dataset/
├── README.md                           # [本文件] 專案總覽與資料集說明
├── requirements.txt                    # 前處理與輔助工具依賴套件
├── configs/
│   ├── traffic_violation.yaml          # YOLOv7 資料集定義設定檔 (路徑與 4 類別)
│   ├── hyp.scratch.custom.yaml         # 消融實驗自訂超參數設定檔
│   └── yolov7-tiny-custom.yaml         # 自訂模型架構檔 (nc=4)
├── scripts/
│   ├── extract_frames.py               # 行車記錄器影片定頻抽幀工具
│   ├── verify_annotations.py           # 標註格式合規性檢查與防呆工具
│   ├── split_dataset.py                # 7:2:1 (Train/Val/Test) 隨機切分工具
│   └── export_metrics.py               # 實驗圖表與指標匯出工具
├── notebooks/
│   └── NCHU_Research_YOLOv7.ipynb      # Google Colab 雲端訓練與評估標準筆記本
├── data/
│   ├── raw/                            # 原始拍攝照片與未標註抽幀圖
│   ├── annotated/                      # 待標註/已標註之有效圖片 (.jpg)
│   ├── labels/                         # X-AnyLabeling 匯出之 YOLO 標籤檔 (.txt)
│   ├── dataset/                        # 切分後之標準 YOLO 格式 (train/val/test)
│   └── dataset.zip                     # 一鍵自動打包產生之雲端訓練壓縮檔
├── sample_videos/                      # 測試用台灣市區行車記錄器短片
└── results/                            # 消融實驗輸出圖表、混淆矩陣與最佳權重備份
```

---

## 🏷️ 標註類別定義 (Target Classes)

本資料集定義台灣道路情境最常見之 **4 大違規停放類別**，統一採用 YOLO 正規化標註格式：

| 類別 ID (`class_id`) | 標籤名稱 (`name`) | 中文類別 | 標註要點與邊界框規範 |
| :---: | :--- | :--- | :--- |
| **0** | `red_line_parking` | 紅線違規停車 | 車體輪廓與路側紅實線緊鄰或重疊之違停汽機車。 |
| **1** | `double_parking` | 並排違規停車 | 車輛停於車道外側，且其同側路旁已有停放車輛（平行並排佔用車道）。 |
| **2** | `yellow_box_parking` | 黃網線臨時暫停 | 靜止停滯於路口或特種出入口黃色網格劃設區內之車輛。 |
| **3** | `sidewalk_parking` | 人行道/騎樓違停 | 違規停放於標線型綠底人行道、實體人行道或人行騎樓內之車輛。 |

* **標籤格式**：每一行代表一個目標，數值經圖片長寬正規化至 `[0, 1]` 區間：
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```

---

## 📊 資料集規模與劃分 (Dataset Scale & Split)

* **目標資料量**：150 ~ 250 張真實道路影像（實地手機街拍 + 行車記錄器抽幀各約 50%）。
* **目標標註框數量**：每類別累積 40 ~ 70 個標註目標（Instances）。
* **切分比例**：
  * **Train (訓練集)**：70%
  * **Val (驗證集)**：20%
  * **Test (測試集)**：10%

---

## 🛠️ 本地資料處理工作流程 (Data Preprocessing Workflow)

### 1. 安裝環境依賴
```bash
pip install -r requirements.txt
```

### 2. 影片抽幀收集
```bash
python scripts/extract_frames.py --video sample_videos/dashcam.mp4 --output data/raw/ --interval 2
```

### 3. 本地影像標註與匯出
推薦使用 **[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)**：
- **開啟圖片目錄**：`data/annotated/`。
- **標註規範**：繪製矩形框緊貼車體輪廓，依序標註 4 大違規類別。
- **匯出 YOLO HBB 標籤**：
  - 點擊 **「導出」$\rightarrow$「YOLO HBB」**。
  - **選取類別定義檔**：選擇 `data/annotated/classes.txt`（作為文字標籤轉譯為 0~3 數字代號之對照字典）。
  - **導出路徑**：選擇 **`data/labels/`**（X-AnyLabeling 預設路徑，避免覆蓋原圖目錄）。
  - **「同時保存圖片？」**：**不用勾選**（原圖已在 `annotated/`，無需重複複製）。

### 4. 標註資料防呆檢驗
執行檢查腳本，自動跨目錄比對圖片與標籤，確認無缺漏、無孤兒檔且數值介於 `0.0 ~ 1.0`：
```bash
python scripts/verify_annotations.py
```
- **核心功能**：自動抓取 `data/annotated/`（圖片）與 `data/labels/`（標籤）進行成對校驗。
- **檢查重點**：確認輸出「0 錯誤」，且終端機顯示各類別數量統計分佈。

### 5. 一鍵資料集切分與自動打包
依 7:2:1 自動隨機切分至 `data/dataset/{train,val,test}`，並**自動生成 `data/dataset.zip`**：
```bash
python scripts/split_dataset.py
```
- **全自動固定邏輯流程**：
  1. 自動讀取 `data/annotated/` 與 `data/labels/` 並成對合併。
  2. 依 7:2:1 比例隨機切分至 `data/dataset/train/`、`val/`、`test/`。
  3. 自動使用純 Python 壓縮打包產出 **`data/dataset.zip`**（跨平台標準，內含標準 `dataset/` 頂層結構，免除手動 zip 操作）。

---

## 🔬 消融實驗規劃 (Ablation Study Matrix)

依據期末報告規定，於 Google Colab 執行 4 組系統化消融對比實驗：

| 組別 | 模型架構 (Model) | 訓練輪數 (Epochs) | Batch | 實驗探討目標 | 評估指標 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **Exp 1** | YOLOv7-tiny | 50 | 16 | 輕量輕量模型基準線 (Baseline) | Params, FLOPs, mAP@0.5 |
| **Exp 2** | YOLOv7 Base | 50 | 8 | 骨幹模型尺度對微小/遮擋目標召回效益 | mAP 增幅 vs. 運算時間 |
| **Exp 3** | 最佳模型 (選 Exp 1/2) | 100 | 16 | 擴增訓練輪數之充分收斂性與過擬合檢視 | Loss 收斂曲線、mAP |
| **Exp 4** | 最佳模型 | 100 | 16 | 自訂超參數 (學習率調校 + Mosaic 增強) | 最終 mAP@[0.5:0.95]、混淆矩陣 |

---

## 🚀 Google Colab 雲端訓練操作簡介

1. 將本地一鍵生成之 **`data/dataset.zip`** 直接上傳至個人 Google Drive 根目錄（或指定目錄）。
2. 開啟 `notebooks/NCHU_Research_YOLOv7.ipynb`。
3. 依序執行儲存格：
   * 掛載 Google Drive
   * Clone 模型倉庫 [nchu-research-method-traffic-violation-model](https://github.com/contemplator/nchu-research-method-traffic-violation-model)
   * 自動解壓縮 `dataset.zip` 並啟動 4 組消融訓練儲存格
   * 運行 `detect.py` 進行行車記錄器影片推論並一鍵打包回傳成果 (`results_summary.zip`)

---

## 🔗 相關專案連結

* 模型庫：[contemplator/nchu-research-method-traffic-violation-model](https://github.com/contemplator/nchu-research-method-traffic-violation-model)
* 上游官方庫：[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
