# NCHU YOLOv7 台灣交通設施偵測 — 瀏覽器端三模型對比展示網站

本專案採用 **Vite + React + TypeScript + Tailwind CSS + ONNX Runtime Web**，所有推論皆於使用者瀏覽器本地執行（優先使用 **WebGPU**，不支援時自動降級為 **WASM**），建置後的 `dist/` 為純靜態檔案，可直接部署至低規格 VPS（2c/2G）而無需任何後端或 GPU 算力。

## 內建三組比較模型

| 模型 | 架構 | 輸入尺寸 | 前處理 | 預設黃金信心門檻 | Test mAP@0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp 5** | YOLOv7-tiny | `640×640` | 原圖 | `0.45` | `43.1%` |
| **Exp 7** | YOLOv7 Base | `640×640` | 原圖 | `0.35` | `43.3%` |
| **Exp 12** | YOLOv7-tiny | `960×960` | LAB-CLAHE | `0.60` | `60.9%` |

## 本地開發與打包部署

```bash
cd web

# 1. 安裝相依套件
npm install

# 2. 本地預覽開發（http://localhost:5173）
npm run dev

# 3. 產生可部署至 VPS 的純靜態檔案（輸出至 web/dist/）
npm run build
```

> **提示**：模型檔案（`.onnx`）位於 `web/public/models/`，可透過專案根目錄的 `scripts/export_web_onnx.py` 重新由 `best.pt` 匯出。
