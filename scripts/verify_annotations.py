#!/usr/bin/env python3
"""
scripts/verify_annotations.py
YOLO 標註格式合規性檢查與防呆驗證工具
支援圖片（data/annotated/）與標註（data/labels/）分開存放或合置於同一目錄。
確保每張圖片皆有對應 .txt、類別 ID 介於 0~3、座標與寬高皆落在 [0.0, 1.0] 區間。
"""

import os
import argparse
from collections import Counter

CLASS_NAMES = {
    0: "red_line_parking",
    1: "double_parking",
    2: "yellow_box_parking",
    3: "sidewalk_parking"
}

def resolve_labels_dir(images_dir: str, labels_dir: str = None) -> str:
    """自動判定標籤存放目錄 (優先 labels_dir -> 預設 data/labels/ -> fallback 到 images_dir)"""
    if labels_dir and os.path.exists(labels_dir):
        return labels_dir
    default_labels = os.path.join(os.path.dirname(images_dir.rstrip('/')), 'labels')
    if os.path.exists(default_labels):
        return default_labels
    return images_dir

def verify_dataset(images_dir: str = "data/annotated/", labels_dir: str = None) -> bool:
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"找不到圖片資料夾: {images_dir}")

    actual_labels_dir = resolve_labels_dir(images_dir, labels_dir)
    if not os.path.exists(actual_labels_dir):
        raise FileNotFoundError(f"找不到標籤資料夾: {actual_labels_dir}")
        
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
    txt_files = [f for f in os.listdir(actual_labels_dir) if f.lower().endswith('.txt') and not f.startswith("classes")]
    
    print(f"🔍 開始驗證標註合規性:")
    print(f"   圖片目錄: {images_dir} (共 {len(images)} 張)")
    print(f"   標籤目錄: {actual_labels_dir} (共 {len(txt_files)} 份)\n")
    
    errors = []
    class_counter = Counter()
    total_boxes = 0
    
    # 1. 檢查每張圖片是否有對應的標註檔
    for img in sorted(images):
        base_name = os.path.splitext(img)[0]
        txt_name = base_name + ".txt"
        txt_path = os.path.join(actual_labels_dir, txt_name)
        
        if not os.path.exists(txt_path):
            errors.append(f"[缺少標註檔] 圖檔 {img} 在 {actual_labels_dir} 沒有找到對應的 {txt_name}")
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if len(lines) == 0:
            errors.append(f"[空標註檔] {txt_name} 是空白檔案（無任何目標框）")
            
        for line_num, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"[格式錯誤] {txt_name} 第 {line_num} 行欄位數不為 5: '{line}'")
                continue
                
            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                errors.append(f"[數值解析錯誤] {txt_name} 第 {line_num} 行無法轉為數值: '{line}'")
                continue
                
            if cls_id not in CLASS_NAMES:
                errors.append(f"[類別 ID 超界] {txt_name} 第 {line_num} 行 class_id={cls_id}，必須介於 0~3")
            else:
                class_counter[cls_id] += 1
                total_boxes += 1
                
            for val_name, val in [("x_center", x_center), ("y_center", y_center), ("w", w), ("h", h)]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"[座標超界] {txt_name} 第 {line_num} 行 {val_name}={val:.4f} 超出 [0, 1] 範圍")
                    
    # 2. 檢查孤兒 txt 檔 (標籤檔找不到對應圖片)
    for txt in sorted(txt_files):
        base_name = os.path.splitext(txt)[0]
        matched_img = any(os.path.exists(os.path.join(images_dir, base_name + ext)) for ext in image_exts)
        if not matched_img:
            errors.append(f"[孤兒標註檔] {txt} 在 {images_dir} 沒有對應的圖片檔案")

    print("📊 類別分佈統計 (Class Distribution):")
    print("-" * 55)
    for cid, name in CLASS_NAMES.items():
        count = class_counter[cid]
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  Class {cid} ({name:<18}): {count:>4} 個目標 ({pct:>5.1f}%)")
    print("-" * 55)
    print(f"  總標註目標數 (Instances): {total_boxes} 個\n")

    if errors:
        print(f"❌ 驗證發現 {len(errors)} 個問題：")
        for err in errors[:20]:
            print("  •", err)
        if len(errors) > 20:
            print(f"  ... 其餘 {len(errors) - 20} 項錯誤已省略")
        return False
    else:
        print("✅ 恭喜！所有標註檔案皆符合 YOLO 標準格式，可以進行資料集切分！\n")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="標註防呆合規性檢查工具")
    parser.add_argument("--images_dir", type=str, default="data/annotated/", help="圖片目錄 (預設: data/annotated/)")
    parser.add_argument("--labels_dir", type=str, default=None, help="標籤目錄 (預設: 自動偵測 data/labels/ 或與圖片同目錄)")
    parser.add_argument("--data_dir", type=str, default=None, help="[舊版相容] 同時包含圖檔與標籤之目錄")
    args = parser.parse_args()
    
    img_dir = args.data_dir if args.data_dir else args.images_dir
    lbl_dir = args.labels_dir
    verify_dataset(img_dir, lbl_dir)
