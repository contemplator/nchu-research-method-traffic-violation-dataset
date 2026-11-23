#!/usr/bin/env python3
"""
scripts/analyze_test_results.py
測試集預測結果與標準答案 (Ground Truth) 系統性比對與可解釋性誤差診斷工具。
支援：
1. 單一實驗 4 維度誤差診斷（TP / FP / FN / Misclassification / Low IoU [0.2 <= IoU < 0.5]）
2. 多實驗整合分析模式（--manifest <json>）：
   - 自動讀取各啟用實驗之 results.txt 繪製 4 象限收斂曲線圖（mAP@0.5, mAP@0.5:0.95, Total Train Loss, Recall）
   - 自動彙整 Master Benchmark 總對照表與 4 維度誤差統計表（輸出 CSV 與終端摘要）
   - 自動挑選難例（優先含 Sign_limit, two_stage_left_turn_box, crosswalk）產出同圖並排推論對比矩陣 (inference_comparison_grid.jpg)
"""

import os
import json
import csv
import glob
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CLASSES = [
    "traffic_light",
    "crosswalk",
    "two_stage_left_turn_box",
    "Sign_follow",
    "Sign_prohibit",
    "Sign_limit",
]

CLASS_COLORS = [
    (46, 204, 113),   # traffic_light - green
    (52, 152, 219),   # crosswalk - blue
    (241, 196, 15),   # two_stage_left_turn_box - yellow
    (155, 89, 182),   # Sign_follow - purple
    (231, 76, 60),    # Sign_prohibit - red
    (230, 126, 34),   # Sign_limit - orange
]


def load_classes(classes_path):
    if classes_path and os.path.exists(classes_path):
        if classes_path.endswith((".yaml", ".yml")):
            try:
                import yaml
                with open(classes_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "names" in data:
                    return list(data["names"])
            except Exception:
                pass
        with open(classes_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        if lines:
            return lines
    return DEFAULT_CLASSES


def box_iou(b1, b2):
    x1_min, y1_min = b1[1] - b1[3] / 2, b1[2] - b1[4] / 2
    x1_max, y1_max = b1[1] + b1[3] / 2, b1[2] + b1[4] / 2
    x2_min, y2_min = b2[1] - b2[3] / 2, b2[2] - b2[4] / 2
    x2_max, y2_max = b2[1] + b2[3] / 2, b2[2] + b2[4] / 2

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h
    union_area = (b1[3] * b1[4]) + (b2[3] * b2[4]) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def read_yolo_txt(file_path, has_conf=False):
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if (has_conf and len(parts) >= 6) else 1.0
            boxes.append([cls_id, xc, yc, w, h, conf])
    return boxes


def analyze_predictions(gt_dir, pred_dir, images_dir=None, classes=None, iou_thresh=0.5, low_iou_thresh=0.2):
    classes = classes or DEFAULT_CLASSES
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    gt_stems = [
        os.path.splitext(os.path.basename(f))[0]
        for f in gt_files
        if not os.path.basename(f).startswith("classes")
    ]

    summary_stats = {
        "total_images": len(gt_stems),
        "total_gt": 0,
        "total_pred": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "pure_fp": 0,
        "class_mismatches": 0,
        "low_iou": 0,
        "per_class": defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0, "fp": 0}),
    }

    image_details = []

    for stem in gt_stems:
        gt_path = os.path.join(gt_dir, f"{stem}.txt")
        pred_path = os.path.join(pred_dir, f"{stem}.txt")

        gt_boxes = read_yolo_txt(gt_path, has_conf=False)
        pred_boxes = read_yolo_txt(pred_path, has_conf=True)
        pred_boxes.sort(key=lambda x: x[5], reverse=True)

        summary_stats["total_gt"] += len(gt_boxes)
        summary_stats["total_pred"] += len(pred_boxes)
        for g in gt_boxes:
            cname = classes[g[0]] if g[0] < len(classes) else str(g[0])
            summary_stats["per_class"][cname]["gt"] += 1

        matched_gt_indices = set()
        img_tp, img_fn, img_fp, img_mismatch, img_low_iou = [], [], [], [], []

        for p in pred_boxes:
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                iou = box_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            p_cname = classes[p[0]] if p[0] < len(classes) else str(p[0])

            if best_iou >= iou_thresh and best_g_idx != -1:
                g = gt_boxes[best_g_idx]
                matched_gt_indices.add(best_g_idx)
                g_cname = classes[g[0]] if g[0] < len(classes) else str(g[0])

                if p[0] == g[0]:
                    summary_stats["tp"] += 1
                    summary_stats["per_class"][p_cname]["tp"] += 1
                    img_tp.append((p_cname, best_iou, p[5]))
                else:
                    summary_stats["class_mismatches"] += 1
                    summary_stats["fp"] += 1
                    summary_stats["per_class"][p_cname]["fp"] += 1
                    img_mismatch.append((g_cname, p_cname, best_iou, p[5]))
            else:
                # 檢查是否屬於定位偏差 (0.2 <= IoU < 0.5)
                max_any_iou = max((box_iou(p, g) for g in gt_boxes), default=0.0)
                summary_stats["fp"] += 1
                summary_stats["per_class"][p_cname]["fp"] += 1
                if low_iou_thresh <= max_any_iou < iou_thresh:
                    summary_stats["low_iou"] += 1
                    img_low_iou.append((p_cname, max_any_iou, p[5]))
                else:
                    summary_stats["pure_fp"] += 1
                    img_fp.append((p_cname, p[5]))

        for g_idx, g in enumerate(gt_boxes):
            if g_idx not in matched_gt_indices:
                g_cname = classes[g[0]] if g[0] < len(classes) else str(g[0])
                summary_stats["fn"] += 1
                summary_stats["per_class"][g_cname]["fn"] += 1
                img_fn.append(g_cname)

        image_details.append({
            "image": stem,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "tp": len(img_tp),
            "fn": len(img_fn),
            "fp": len(img_fp) + len(img_mismatch) + len(img_low_iou),
            "low_iou": len(img_low_iou),
            "tp_details": img_tp,
            "fn_details": img_fn,
            "fp_details": img_fp,
            "mismatch_details": img_mismatch,
            "low_iou_details": img_low_iou,
        })

    return summary_stats, image_details


def print_readable_report(summary_stats, image_details):
    print("=" * 72)
    print("📊 YOLO 測試集系統性比對與 4 維度誤差診斷報告")
    print("=" * 72)
    print(f"• 測試樣本總數: {summary_stats['total_images']} 張")
    print(f"• 真實目標總數 (Ground Truth): {summary_stats['total_gt']} 個")
    print(f"• 模型檢測總數 (Predictions) : {summary_stats['total_pred']} 個")
    print("-" * 72)

    tp_pct = (summary_stats["tp"] / summary_stats["total_gt"] * 100) if summary_stats["total_gt"] > 0 else 0
    fn_pct = (summary_stats["fn"] / summary_stats["total_gt"] * 100) if summary_stats["total_gt"] > 0 else 0
    fp_pct = (summary_stats["fp"] / summary_stats["total_pred"] * 100) if summary_stats["total_pred"] > 0 else 0

    print(f"✅ 成功辨識 (True Positives, TP)      : {summary_stats['tp']:>3} 個 ({tp_pct:5.1f}% 召回)")
    print(f"⚠️ 漏抓目標 (False Negatives, FN)     : {summary_stats['fn']:>3} 個 ({fn_pct:5.1f}% 漏檢)")
    print(f"❓ 誤判總數 (Total False Positives)   : {summary_stats['fp']:>3} 個 ({fp_pct:5.1f}% 誤報)")
    print(f"   ├─ 背景假陽性 (Pure FP)            : {summary_stats['pure_fp']:>3} 個")
    print(f"   ├─ 類別混淆 (Misclassification)    : {summary_stats['class_mismatches']:>3} 個")
    print(f"   └─ 邊界偏差 (Low IoU 0.2~0.5)      : {summary_stats['low_iou']:>3} 個")
    print("-" * 72)

    print("\n🔍 各類別細部統計 (Class Breakdown):")
    header = f"{'類別名稱 (Class)':<24} | {'GT':<4} | {'TP':<5} | {'FN':<5} | {'FP':<5} | {'Recall':<7} | {'Precision':<9}"
    print(header)
    print("-" * len(header))
    for cname, stats in sorted(summary_stats["per_class"].items()):
        gt, tp, fn, fp = stats["gt"], stats["tp"], stats["fn"], stats["fp"]
        recall = (tp / gt * 100) if gt > 0 else 0.0
        prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        print(f"{cname:<24} | {gt:>4} | {tp:>5} | {fn:>5} | {fp:>5} | {recall:>6.1f}% | {prec:>8.1f}%")

    print(f"\n📸 測試集 {summary_stats['total_images']} 張影像逐張偵測明細:")
    print("-" * 72)
    for img in image_details:
        parts = []
        if img["tp_details"]:
            tps = [f"{c}({conf:.2f})" for c, _, conf in img["tp_details"]]
            parts.append(f"✅抓到({len(tps)}): {', '.join(tps)}")
        else:
            parts.append("✅抓到(0)")
        if img["mismatch_details"]:
            mm = [f"真值{g}->預測{p}({conf:.2f})" for g, p, _, conf in img["mismatch_details"]]
            parts.append(f"🔄混淆({len(mm)}): {', '.join(mm)}")
        if img["low_iou_details"]:
            li = [f"{c}(IoU={iou:.2f})" for c, iou, _ in img["low_iou_details"]]
            parts.append(f"📐LowIoU({len(li)}): {', '.join(li)}")
        if img["fp_details"]:
            fps = [f"{c}({conf:.2f})" for c, conf in img["fp_details"]]
            parts.append(f"❓誤判({len(fps)}): {', '.join(fps)}")
        if img["fn_details"]:
            parts.append(f"⚠️漏抓({len(img['fn_details'])}): {', '.join(img['fn_details'])}")
        print(f"• {img['image']} (真值:{img['gt_count']} | 預測:{img['pred_count']})\n    " + " ｜ ".join(parts))
    print("=" * 72)


def parse_yolov7_results_txt(results_txt_path):
    """解析 YOLOv7 results.txt，回傳各 epoch 指標字典列表。"""
    epochs, total_loss, recall, map50, map50_95 = [], [], [], [], []
    if not results_txt_path or not os.path.exists(results_txt_path):
        return None
    with open(results_txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            try:
                ep_str = parts[0].split("/")[0]
                ep = int(ep_str) + 1
                t_loss = float(parts[5])
                r_val = float(parts[9])
                m50 = float(parts[10])
                m95 = float(parts[11])
            except (ValueError, IndexError):
                continue
            epochs.append(ep if ep > 0 else idx)
            total_loss.append(t_loss)
            recall.append(r_val)
            map50.append(m50)
            map50_95.append(m95)
    if not epochs:
        return None
    return {
        "epochs": epochs,
        "total_loss": total_loss,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
    }


def plot_convergence_quadrants(experiments, output_path):
    """繪製 4 象限收斂曲線圖：mAP@0.5, mAP@0.5:0.95, Total Train Loss, Recall"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    panels = [
        (axes[0, 0], "map50", "mAP@0.5 Convergence", "mAP@0.5"),
        (axes[0, 1], "map50_95", "mAP@0.5:0.95 Convergence", "mAP@0.5:0.95"),
        (axes[1, 0], "total_loss", "Total Train Loss Convergence", "Loss"),
        (axes[1, 1], "recall", "Validation Recall Convergence", "Recall"),
    ]

    plotted = 0
    for exp in experiments:
        parsed = parse_yolov7_results_txt(exp.get("results_txt"))
        if not parsed:
            continue
        plotted += 1
        label = exp.get("label", exp.get("id", "exp"))
        for ax, key, _, _ in panels:
            ax.plot(parsed["epochs"], parsed[key], linewidth=1.8, label=label)

    for ax, _, title, ylabel in panels:
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        if plotted > 0:
            ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 已輸出 4 象限收斂對照圖: {output_path} (共 {plotted} 組實驗)")


def find_image_path(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def draw_boxes_on_image(img_path, boxes, classes, title, is_gt=False):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for b in boxes:
        cls_id, xc, yc, bw, bh = int(b[0]), b[1], b[2], b[3], b[4]
        conf = b[5] if len(b) > 5 else 1.0
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w - 1, int((xc + bw / 2) * w))
        y2 = min(h - 1, int((yc + bh / 2) * h))
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cname = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        label = cname if is_gt else f"{cname} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color, font=font)

    # 頂部標題列
    banner_h = 28
    canvas = Image.new("RGB", (w, h + banner_h), (25, 25, 25))
    canvas.paste(img, (0, banner_h))
    d2 = ImageDraw.Draw(canvas)
    d2.text((8, 7), title, fill=(255, 255, 255), font=font)
    return canvas


def select_hard_stems(gt_dir, classes, max_images=5):
    """優先挑選含 Sign_limit (5), two_stage_left_turn_box (2), crosswalk (1) 或框數較多的測試圖。"""
    target_cls = {5: 10, 2: 6, 1: 3}
    scored = []
    for gt_file in sorted(glob.glob(os.path.join(gt_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(gt_file))[0]
        if stem.startswith("classes"):
            continue
        boxes = read_yolo_txt(gt_file, has_conf=False)
        score = sum(target_cls.get(b[0], 1) for b in boxes)
        scored.append((score, len(boxes), stem))
    scored.sort(reverse=True)
    return [s[2] for s in scored[:max_images]]


def generate_comparison_grid(experiments, gt_dir, images_dir, classes, output_path, max_images=5):
    """產出所有啟用實驗的同圖並排推論對比矩陣 (Ground Truth + 各實驗預測)。"""
    valid_exps = [e for e in experiments if e.get("detect_labels_dir") and os.path.exists(e["detect_labels_dir"])]
    if not valid_exps or not os.path.exists(images_dir):
        print("⚠️ 略過推論對比矩陣：找不到可用的 detect labels 或測試影像目錄。")
        return

    stems = select_hard_stems(gt_dir, classes, max_images=max_images)
    if not stems:
        print("⚠️ 略過推論對比矩陣：測試集無樣本。")
        return

    tile_w, tile_h = 480, 300
    rows = []
    for stem in stems:
        img_path = find_image_path(images_dir, stem)
        if not img_path:
            continue
        gt_boxes = read_yolo_txt(os.path.join(gt_dir, f"{stem}.txt"), has_conf=False)
        row_tiles = [
            draw_boxes_on_image(img_path, gt_boxes, classes, f"GT | {stem}", is_gt=True).resize((tile_w, tile_h))
        ]
        for exp in valid_exps:
            pred_path = os.path.join(exp["detect_labels_dir"], f"{stem}.txt")
            pred_boxes = read_yolo_txt(pred_path, has_conf=True)
            exp_img_dir = exp.get("images_dir") or images_dir
            exp_img_path = find_image_path(exp_img_dir, stem) or img_path
            title = exp.get("short_label", exp.get("id", "exp"))
            row_tiles.append(
                draw_boxes_on_image(exp_img_path, pred_boxes, classes, title, is_gt=False).resize((tile_w, tile_h))
            )
        rows.append(row_tiles)

    if not rows:
        return

    ncols = len(rows[0])
    nrows = len(rows)
    grid = Image.new("RGB", (ncols * tile_w, nrows * tile_h), (15, 15, 15))
    for r_idx, row_tiles in enumerate(rows):
        for c_idx, tile in enumerate(row_tiles):
            grid.paste(tile, (c_idx * tile_w, r_idx * tile_h))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid.save(output_path, quality=92)
    print(f"🖼️ 已輸出同圖並排推論對比矩陣: {output_path} ({nrows} 張難例 × {ncols} 欄)")


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def compute_composite_scores(rows, acc_weight=0.6, speed_weight=0.4):
    """
    計算準確度與速度的綜合評分 (Composite Score) 與 Pareto 前緣標記：
    - Acc_Norm   = 0.5 * (mAP@0.5 / max_mAP@0.5) + 0.5 * (F1 / max_F1)
    - Speed_Norm = FPS / max_FPS
    - Composite_Score = acc_weight * Acc_Norm + speed_weight * Speed_Norm
    """
    valid_points = []
    for idx, r in enumerate(rows):
        m50 = _safe_float(r.get("mAP@0.5"))
        f1 = _safe_float(r.get("F1"))
        fps = _safe_float(r.get("FPS"))
        if m50 is not None and f1 is not None and fps is not None and fps > 0:
            valid_points.append((idx, m50, f1, fps))

    max_m50 = max((p[1] for p in valid_points), default=0.0)
    max_f1 = max((p[2] for p in valid_points), default=0.0)
    max_fps = max((p[3] for p in valid_points), default=0.0)

    # 計算 (mAP@0.5, FPS) Pareto Frontier
    pareto_indices = set()
    for idx_i, m50_i, _, fps_i in valid_points:
        dominated = False
        for idx_j, m50_j, _, fps_j in valid_points:
            if idx_i == idx_j:
                continue
            if m50_j >= m50_i and fps_j >= fps_i and (m50_j > m50_i or fps_j > fps_i):
                dominated = True
                break
        if not dominated:
            pareto_indices.add(idx_i)

    for idx, r in enumerate(rows):
        m50 = _safe_float(r.get("mAP@0.5"))
        f1 = _safe_float(r.get("F1"))
        fps = _safe_float(r.get("FPS"))
        if m50 is not None and f1 is not None and fps is not None and max_m50 > 0 and max_f1 > 0 and max_fps > 0:
            acc_norm = 0.5 * (m50 / max_m50) + 0.5 * (f1 / max_f1)
            speed_norm = fps / max_fps
            comp = acc_weight * acc_norm + speed_weight * speed_norm
            r["Composite_Score"] = f"{comp:.3f}"
            r["Pareto"] = "Yes" if idx in pareto_indices else "No"
        else:
            r["Composite_Score"] = "-"
            r["Pareto"] = "-"
    return rows


def plot_accuracy_speed_tradeoff(rows, output_path):
    """繪製 Accuracy vs. Speed (FPS vs mAP@0.5 Pareto 圖 + 綜合評分排行圖)。"""
    import matplotlib.pyplot as plt

    valid = []
    for r in rows:
        m50 = _safe_float(r.get("mAP@0.5"))
        f1 = _safe_float(r.get("F1"))
        fps = _safe_float(r.get("FPS"))
        comp = _safe_float(r.get("Composite_Score"))
        if m50 is not None and f1 is not None and fps is not None and comp is not None:
            valid.append({
                "label": r.get("Experiment", ""),
                "arch": r.get("Arch", ""),
                "m50": m50,
                "f1": f1,
                "fps": fps,
                "comp": comp,
                "pareto": r.get("Pareto") == "Yes",
            })

    if not valid:
        print("⚠️ 略過 Accuracy-Speed 權衡圖：無可用的數值資料。")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)

    # 左圖：FPS vs mAP@0.5 散佈圖 + Pareto 前緣
    for item in valid:
        is_tiny = "tiny" in item["arch"].lower()
        color = "#1f77b4" if is_tiny else "#d95f02"
        marker = "o" if is_tiny else "s"
        edge = "#000000" if item["pareto"] else "white"
        size = 140 if item["pareto"] else 95
        ax1.scatter(item["fps"], item["m50"], c=color, marker=marker, s=size, edgecolors=edge, linewidths=1.5, zorder=4)
        short = item["label"].split(":")[0] if ":" in item["label"] else item["label"]
        ax1.annotate(
            f"{short}\n({item['m50']:.3f}, {item['fps']:.0f}fps)",
            (item["fps"], item["m50"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            fontweight="bold" if item["pareto"] else "normal",
        )

    # 繪製 Pareto Frontier 連線
    pareto_pts = sorted([v for v in valid if v["pareto"]], key=lambda x: x["fps"])
    if len(pareto_pts) >= 2:
        ax1.plot(
            [p["fps"] for p in pareto_pts],
            [p["m50"] for p in pareto_pts],
            linestyle="--",
            color="#2ca02c",
            linewidth=1.8,
            alpha=0.85,
            label="Pareto Frontier",
            zorder=3,
        )

    # 即時推論門檻參考線
    ax1.axvline(30, color="#888888", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.text(31, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0.02, "30 FPS", fontsize=8, color="#666666")
    ax1.axvline(60, color="#888888", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.text(61, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0.02, "60 FPS", fontsize=8, color="#666666")

    # 圖例代理
    ax1.scatter([], [], c="#1f77b4", marker="o", s=95, label="YOLOv7-tiny")
    ax1.scatter([], [], c="#d95f02", marker="s", s=95, label="YOLOv7 Base")
    ax1.set_title("Accuracy vs. Inference Speed (Test Set, 47 imgs)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Inference Speed (FPS)", fontsize=11)
    ax1.set_ylabel("mAP@0.5", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=9)

    # 右圖：綜合評分排行橫條圖
    ranked = sorted(valid, key=lambda x: x["comp"])
    labels = [x["label"] for x in ranked]
    scores = [x["comp"] for x in ranked]
    colors = ["#1f77b4" if "tiny" in x["arch"].lower() else "#d95f02" for x in ranked]
    bars = ax2.barh(labels, scores, color=colors, alpha=0.88, edgecolor="#333333", linewidth=0.6)
    for bar, item in zip(bars, ranked):
        star = " ★" if item["pareto"] else ""
        ax2.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{item['comp']:.3f}{star}",
            va="center",
            fontsize=8.5,
            fontweight="bold" if item["pareto"] else "normal",
        )
    ax2.set_xlim(0, min(1.18, max(scores) + 0.16))
    ax2.set_title("Composite Score Ranking (0.6×Accuracy + 0.4×Speed)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Composite Score (★ = Pareto Optimal)", fontsize=11)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"⚡ 已輸出 Accuracy vs. Speed 權衡分析圖: {output_path}")


def run_benchmark_manifest(manifest_path, output_dir, classes, iou_thresh=0.5, max_grid_images=5):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    experiments = manifest.get("experiments", [])
    gt_dir = manifest.get("gt_dir", "data/dataset/test/labels")
    images_dir = manifest.get("images_dir", "data/dataset/test/images")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 4 象限收斂對照圖
    plot_convergence_quadrants(experiments, os.path.join(output_dir, "convergence_quadrants.png"))

    # 2. Master Benchmark + 4 維度誤差表
    rows = []
    for exp in experiments:
        detect_dir = exp.get("detect_labels_dir")
        err_stats = {
            "tp": "-", "fp": "-", "fn": "-",
            "misclassification": "-", "low_iou": "-",
        }
        if detect_dir and os.path.exists(detect_dir):
            exp_gt_dir = exp.get("gt_dir") or gt_dir
            s, _ = analyze_predictions(exp_gt_dir, detect_dir, images_dir, classes, iou_thresh=iou_thresh)
            err_stats = {
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "misclassification": s["class_mismatches"],
                "low_iou": s["low_iou"],
            }

        rows.append({
            "Experiment": exp.get("label", exp.get("id")),
            "Arch": exp.get("arch", ""),
            "ImgSize": exp.get("img_size", ""),
            "Batch": exp.get("batch", ""),
            "Epochs": exp.get("epochs", ""),
            "BestConf": exp.get("best_conf", "-"),
            "mAP@0.5": exp.get("map50", "-"),
            "mAP@0.5:0.95": exp.get("map50_95", "-"),
            "Precision": exp.get("precision", "-"),
            "Recall": exp.get("recall", "-"),
            "F1": exp.get("f1", "-"),
            "ms/img": exp.get("ms_per_img", "-"),
            "FPS": exp.get("fps", "-"),
            "TP": err_stats["tp"],
            "FP": err_stats["fp"],
            "FN": err_stats["fn"],
            "Misclassification": err_stats["misclassification"],
            "Low_IoU": err_stats["low_iou"],
        })

    compute_composite_scores(rows)

    csv_path = os.path.join(output_dir, "master_benchmark.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n📋 已輸出 Master Benchmark 總對照表: {csv_path}")
        print("-" * 134)
        print(
            f"{'Experiment':<26} | {'BestConf':>8} | {'mAP50':>7} | {'mAP95':>7} | "
            f"{'P':>6} | {'R':>6} | {'F1':>6} | {'FPS':>6} | "
            f"{'TP':>4} | {'FP':>4} | {'FN':>4} | {'Mis':>4} | {'LowIoU':>6} | {'Score':>6} | {'Pareto':>6}"
        )
        print("-" * 134)
        for r in rows:
            print(
                f"{str(r['Experiment']):<26} | {str(r['BestConf']):>8} | {str(r['mAP@0.5']):>7} | "
                f"{str(r['mAP@0.5:0.95']):>7} | {str(r['Precision']):>6} | {str(r['Recall']):>6} | "
                f"{str(r['F1']):>6} | {str(r['FPS']):>6} | {str(r['TP']):>4} | {str(r['FP']):>4} | "
                f"{str(r['FN']):>4} | {str(r['Misclassification']):>4} | {str(r['Low_IoU']):>6} | "
                f"{str(r['Composite_Score']):>6} | {str(r['Pareto']):>6}"
            )
        print("-" * 134)

        # 印出綜合評分排行摘要
        ranked_rows = [r for r in rows if _safe_float(r.get("Composite_Score")) is not None]
        ranked_rows.sort(key=lambda x: float(x["Composite_Score"]), reverse=True)
        if ranked_rows:
            print("\n🏆 準確度 × 速度 綜合評分排行 (Composite = 0.6×Accuracy + 0.4×Speed):")
            for rank, r in enumerate(ranked_rows, start=1):
                tag = " ★ Pareto" if r.get("Pareto") == "Yes" else ""
                print(
                    f"  #{rank:<2} {r['Experiment']:<30} [{r['Arch']:<11}] "
                    f"mAP50={r['mAP@0.5']} | F1={r['F1']} | FPS={r['FPS']} | Score={r['Composite_Score']}{tag}"
                )

    # 3. Accuracy vs. Speed 權衡圖
    plot_accuracy_speed_tradeoff(rows, os.path.join(output_dir, "accuracy_speed_tradeoff.png"))

    # 4. 橫向推論對比矩陣
    generate_comparison_grid(
        experiments,
        gt_dir,
        images_dir,
        classes,
        os.path.join(output_dir, "inference_comparison_grid.jpg"),
        max_images=max_grid_images,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 測試集可解釋性誤差診斷與多實驗基準分析工具")
    parser.add_argument("--manifest", type=str, default=None, help="多實驗整合分析 JSON 清單路徑")
    parser.add_argument("--output-dir", type=str, default="results/benchmark", help="多實驗輸出目錄")
    parser.add_argument("--max-grid-images", type=int, default=5, help="對比矩陣挑選難例張數")
    parser.add_argument("--gt", type=str, default="data/dataset/test/labels", help="真實測試集標籤目錄")
    parser.add_argument("--pred", type=str, default="runs/detect/detect_test_images/labels", help="模型預測標籤目錄")
    parser.add_argument("--images", type=str, default="data/dataset/test/images", help="測試集原圖目錄")
    parser.add_argument("--classes", type=str, default="data/dataset/classes.txt", help="類別對照檔")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 判定門檻 (預設 0.5)")
    args = parser.parse_args()

    classes = load_classes(args.classes)
    if args.manifest:
        run_benchmark_manifest(
            args.manifest,
            args.output_dir,
            classes,
            iou_thresh=args.iou,
            max_grid_images=args.max_grid_images,
        )
    else:
        stats, details = analyze_predictions(args.gt, args.pred, args.images, classes, args.iou)
        print_readable_report(stats, details)
