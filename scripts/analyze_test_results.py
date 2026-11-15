#!/usr/bin/env python3
"""
scripts/analyze_test_results.py
測試集預測結果與標準答案 (Ground Truth) 系統性比對與可解釋性誤差診斷工具。
自動比對 test/labels/ 與 detect.py 輸出之 labels/，分類：
- TP (True Positive): 成功抓到 (IoU >= 0.5 且類別正確)
- FN (False Negative): 漏抓目標 (未檢出之真實物件)
- FP (False Positive): 誤判/虛警 (無對應真值之預測框)
- Class Error: 類別混淆 (位置重疊但分類錯誤)
"""

import os
import argparse
import glob
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CLASSES = [
    "traffic_light",
    "crosswalk",
    "two_stage_left_turn_box",
    "Sign_follow",
    "Sign_prohibit",
    "Sign_limit"
]

def load_classes(classes_path):
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            return lines
    return DEFAULT_CLASSES

def box_iou(b1, b2):
    # b: [cls, xc, yc, w, h]
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
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if (has_conf and len(parts) >= 6) else 1.0
            boxes.append([cls_id, xc, yc, w, h, conf])
    return boxes

def analyze_predictions(gt_dir, pred_dir, images_dir=None, classes=None, iou_thresh=0.5, output_dir=None):
    classes = classes or DEFAULT_CLASSES
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    gt_stems = [os.path.splitext(os.path.basename(f))[0] for f in gt_files if not f.endswith("classes.txt")]

    summary_stats = {
        "total_images": len(gt_stems),
        "total_gt": 0,
        "total_pred": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "class_mismatches": 0,
        "per_class": defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0, "fp": 0})
    }

    image_details = []

    for stem in gt_stems:
        gt_path = os.path.join(gt_dir, f"{stem}.txt")
        pred_path = os.path.join(pred_dir, f"{stem}.txt")

        gt_boxes = read_yolo_txt(gt_path, has_conf=False)
        pred_boxes = read_yolo_txt(pred_path, has_conf=True)
        # sort predictions by confidence descending
        pred_boxes.sort(key=lambda x: x[5], reverse=True)

        summary_stats["total_gt"] += len(gt_boxes)
        summary_stats["total_pred"] += len(pred_boxes)
        for g in gt_boxes:
            cname = classes[g[0]] if g[0] < len(classes) else str(g[0])
            summary_stats["per_class"][cname]["gt"] += 1

        matched_gt_indices = set()
        matched_pred_indices = set()

        img_tp = []
        img_fn = []
        img_fp = []
        img_mismatch = []

        # 1. Match predictions to ground truth
        for p_idx, p in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                iou = box_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_thresh and best_g_idx != -1:
                g = gt_boxes[best_g_idx]
                matched_gt_indices.add(best_g_idx)
                matched_pred_indices.add(p_idx)
                p_cname = classes[p[0]] if p[0] < len(classes) else str(p[0])
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
                p_cname = classes[p[0]] if p[0] < len(classes) else str(p[0])
                summary_stats["fp"] += 1
                summary_stats["per_class"][p_cname]["fp"] += 1
                img_fp.append((p_cname, p[5]))

        # 2. Unmatched ground truths are False Negatives (FN)
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
            "fp": len(img_fp) + len(img_mismatch),
            "tp_details": img_tp,
            "fn_details": img_fn,
            "fp_details": img_fp,
            "mismatch_details": img_mismatch
        })

    return summary_stats, image_details

def print_readable_report(summary_stats, image_details):
    print("=" * 70)
    print("📊 YOLO 測試集系統性比對與可解釋性誤差診斷報告")
    print("=" * 70)
    print(f"• 測試樣本總數: {summary_stats['total_images']} 張")
    print(f"• 真實目標總數 (Ground Truth): {summary_stats['total_gt']} 個")
    print(f"• 模型檢測總數 (Predictions) : {summary_stats['total_pred']} 個")
    print("-" * 70)
    
    tp_pct = (summary_stats['tp'] / summary_stats['total_gt'] * 100) if summary_stats['total_gt'] > 0 else 0
    fn_pct = (summary_stats['fn'] / summary_stats['total_gt'] * 100) if summary_stats['total_gt'] > 0 else 0
    fp_pct = (summary_stats['fp'] / summary_stats['total_pred'] * 100) if summary_stats['total_pred'] > 0 else 0

    print(f"✅ 成功辨識 (True Positives, TP) : {summary_stats['tp']:>3} 個 ({tp_pct:5.1f}% 召回)")
    print(f"⚠️ 漏抓目標 (False Negatives, FN): {summary_stats['fn']:>3} 個 ({fn_pct:5.1f}% 漏檢)")
    print(f"❓ 誤判虛警 (False Positives, FP): {summary_stats['fp']:>3} 個 ({fp_pct:5.1f}% 誤報)")
    if summary_stats['class_mismatches'] > 0:
        print(f"🔄 類別混淆 (Class Mismatch)   : {summary_stats['class_mismatches']:>3} 個")
    print("-" * 70)

    print("\n🔍 各類別細部統計 (Class Breakdown):")
    header = f"{'類別名稱 (Class)':<24} | {'GT':<4} | {'TP (抓到)':<9} | {'FN (漏抓)':<9} | {'FP (誤判)':<9} | {'Recall':<7} | {'Precision':<9}"
    print(header)
    print("-" * len(header))
    for cname, stats in sorted(summary_stats['per_class'].items()):
        gt = stats['gt']
        tp = stats['tp']
        fn = stats['fn']
        fp = stats['fp']
        recall = (tp / gt * 100) if gt > 0 else 0.0
        prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        print(f"{cname:<24} | {gt:>4} | {tp:>9} | {fn:>9} | {fp:>9} | {recall:>6.1f}% | {prec:>8.1f}%")

    # List top images with FN (missed) or FP (false alarms)
    troubles = [img for img in image_details if img['fn'] > 0 or img['fp'] > 0]
    print(f"\n⚠️ 需要注意的重點分析樣本 (共 {len(troubles)} 張有漏抓或誤判):")
    print("-" * 70)
    for t in troubles[:8]:
        issues = []
        if t['fn_details']:
            issues.append(f"漏抓: {', '.join(t['fn_details'])}")
        if t['fp_details']:
            fps = [f"{c}(conf {conf:.2f})" for c, conf in t['fp_details']]
            issues.append(f"誤判: {', '.join(fps)}")
        if t['mismatch_details']:
            mm = [f"真值{g}->預測{p}" for g, p, iou, conf in t['mismatch_details']]
            issues.append(f"類別混淆: {', '.join(mm)}")
        print(f"• {t['image']}: {' | '.join(issues)}")
    if len(troubles) > 8:
        print(f"  ... 其餘 {len(troubles) - 8} 張已省略 (完整成果已儲存)")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 測試集可解釋性誤差診斷分析工具")
    parser.add_argument("--gt", type=str, default="data/dataset/test/labels", help="真實測試集標籤目錄")
    parser.add_argument("--pred", type=str, default="runs/detect/detect_test_images/labels", help="模型預測標籤目錄")
    parser.add_argument("--images", type=str, default="data/dataset/test/images", help="測試集原圖目錄")
    parser.add_argument("--classes", type=str, default="data/dataset/classes.txt", help="類別對照檔")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 判定門檻 (預設 0.5)")
    args = parser.parse_args()

    classes = load_classes(args.classes)
    stats, details = analyze_predictions(args.gt, args.pred, args.images, classes, args.iou)
    print_readable_report(stats, details)
