#!/usr/bin/env python3
"""
scripts/verify_annotations.py
YOLO 標註格式合規性檢查與防呆驗證工具
支援：
1. 單一目錄驗證（--images_dir / --labels_dir）
2. 切分後資料集全 split 驗證（--dataset_dir data/dataset，自動檢查 train / val / test）
3. 合法空標註檔（無目標負樣本 Negative Samples）統計與類別分佈統計
"""

import os
import argparse
from collections import Counter

DEFAULT_CLASS_NAMES = {
    0: "traffic_light",
    1: "crosswalk",
    2: "two_stage_left_turn_box",
    3: "Sign_follow",
    4: "Sign_prohibit",
    5: "Sign_limit",
}


def load_class_names(labels_dir: str, images_dir: str, dataset_dir: str = None) -> dict:
    """從 classes.txt 讀取類別對照表，若無則使用標準 6 類別預設值"""
    candidates = [labels_dir, images_dir, dataset_dir]
    if labels_dir:
        candidates.append(os.path.dirname(labels_dir.rstrip("/")))
    for d in candidates:
        if d:
            cpath = os.path.join(d, "classes.txt")
            if os.path.exists(cpath):
                with open(cpath, "r", encoding="utf-8") as f:
                    cnames = [l.strip() for l in f if l.strip()]
                if cnames:
                    return {i: name for i, name in enumerate(cnames)}
    return dict(DEFAULT_CLASS_NAMES)


def resolve_labels_dir(images_dir: str, labels_dir: str = None) -> str:
    """自動判定標籤存放目錄 (labels_dir -> 同層 labels/ -> 上層 labels/ -> fallback images_dir)"""
    if labels_dir and os.path.exists(labels_dir):
        return labels_dir
    clean_img = images_dir.rstrip("/")
    # e.g. data/dataset/train/images -> data/dataset/train/labels
    sibling_labels = os.path.join(os.path.dirname(clean_img), "labels")
    if os.path.exists(sibling_labels):
        return sibling_labels
    return images_dir


def verify_split(
    images_dir: str,
    labels_dir: str = None,
    class_names: dict = None,
    split_name: str = None,
) -> tuple[bool, dict]:
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"找不到圖片資料夾: {images_dir}")

    actual_labels_dir = resolve_labels_dir(images_dir, labels_dir)
    if not os.path.exists(actual_labels_dir):
        raise FileNotFoundError(f"找不到標籤資料夾: {actual_labels_dir}")

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
    txt_files = [
        f
        for f in os.listdir(actual_labels_dir)
        if f.lower().endswith(".txt") and not f.startswith("classes")
    ]

    class_names = class_names or load_class_names(actual_labels_dir, images_dir)
    title = f"[{split_name}] " if split_name else ""
    print(f"🔍 {title}開始驗證標註合規性:")
    print(f"   圖片目錄: {images_dir} (共 {len(images)} 張)")
    print(f"   標籤目錄: {actual_labels_dir} (共 {len(txt_files)} 份)")
    print(f"   類別數量: {len(class_names)} ({', '.join(class_names.values())})\n")

    errors = []
    class_counter = Counter()
    total_boxes = 0
    negative_samples = []

    for img in sorted(images):
        base_name = os.path.splitext(img)[0]
        txt_name = base_name + ".txt"
        txt_path = os.path.join(actual_labels_dir, txt_name)

        if not os.path.exists(txt_path):
            errors.append(f"[缺少標註檔] 圖檔 {img} 在 {actual_labels_dir} 沒有找到對應的 {txt_name}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            negative_samples.append(img)
            continue

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

            if cls_id not in class_names:
                errors.append(
                    f"[類別 ID 超界] {txt_name} 第 {line_num} 行 class_id={cls_id}，必須介於 0~{len(class_names)-1}"
                )
            else:
                class_counter[cls_id] += 1
                total_boxes += 1

            for val_name, val in [("x_center", x_center), ("y_center", y_center), ("w", w), ("h", h)]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"[座標超界] {txt_name} 第 {line_num} 行 {val_name}={val:.4f} 超出 [0, 1] 範圍")

    for txt in sorted(txt_files):
        base_name = os.path.splitext(txt)[0]
        matched_img = any(os.path.exists(os.path.join(images_dir, base_name + ext)) for ext in image_exts)
        if not matched_img:
            errors.append(f"[孤兒標註檔] {txt} 在 {images_dir} 沒有對應的圖片檔案")

    neg_pct = (len(negative_samples) / len(images) * 100) if images else 0.0
    print(f"📊 {title}類別分佈統計 (Class Distribution):")
    print("-" * 62)
    for cid, name in class_names.items():
        count = class_counter[cid]
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0.0
        print(f"  Class {cid} ({name:<24}): {count:>4} 個目標 ({pct:>5.1f}%)")
    print("-" * 62)
    print(f"  總標註目標數 (Instances) : {total_boxes} 個")
    print(f"  無目標負樣本 (Negatives) : {len(negative_samples)} 張 ({neg_pct:.1f}%)\n")

    if errors:
        print(f"❌ {title}驗證發現 {len(errors)} 個問題：")
        for err in errors[:20]:
            print("  •", err)
        if len(errors) > 20:
            print(f"  ... 其餘 {len(errors) - 20} 項錯誤已省略")
        ok = False
    else:
        print(f"✅ {title}所有標註檔案皆符合 YOLO 標準格式！\n")
        ok = True

    summary = {
        "images": len(images),
        "boxes": total_boxes,
        "negatives": len(negative_samples),
        "class_counter": class_counter,
        "errors": len(errors),
    }
    return ok, summary


def verify_dataset_splits(dataset_dir: str) -> bool:
    """驗證切分後的 train / val / test 三個 split 并輸出總表"""
    splits = ["train", "val", "test"]
    class_names = load_class_names(None, None, dataset_dir=dataset_dir)
    all_ok = True
    split_summaries = {}

    print("=" * 68)
    print(f"🛡️ 資料品質防線驗證：{dataset_dir}")
    print("=" * 68)

    for split in splits:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if os.path.exists(img_dir):
            ok, summary = verify_split(img_dir, lbl_dir, class_names=class_names, split_name=split.upper())
            all_ok = all_ok and ok
            split_summaries[split] = summary
        else:
            print(f"⚠️ 找不到 {img_dir}，略過 {split}")

    if split_summaries:
        total_imgs = sum(s["images"] for s in split_summaries.values())
        total_boxes = sum(s["boxes"] for s in split_summaries.values())
        total_negs = sum(s["negatives"] for s in split_summaries.values())
        print("=" * 68)
        print("📋 各 Split 總覽:")
        for split, s in split_summaries.items():
            neg_pct = (s["negatives"] / s["images"] * 100) if s["images"] else 0.0
            print(
                f"  - {split:<5}: {s['images']:>3} 張影像 | {s['boxes']:>4} 框 | "
                f"負樣本 {s['negatives']} 張 ({neg_pct:.1f}%) | 錯誤 {s['errors']} 筆"
            )
        print(f"  - 合計 : {total_imgs:>3} 張影像 | {total_boxes:>4} 框 | 負樣本 {total_negs} 張")
        print("=" * 68 + "\n")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="標註防呆合規性檢查工具")
    parser.add_argument("--dataset_dir", type=str, default=None, help="切分後資料集根目錄 (含 train/val/test)")
    parser.add_argument("--images_dir", type=str, default="data/annotated/", help="圖片目錄 (預設: data/annotated/)")
    parser.add_argument("--labels_dir", type=str, default=None, help="標籤目錄 (預設: 自動偵測)")
    parser.add_argument("--data_dir", type=str, default=None, help="[舊版相容] 同時包含圖檔與標籤之目錄")
    args = parser.parse_args()

    if args.dataset_dir:
        ok = verify_dataset_splits(args.dataset_dir)
    else:
        img_dir = args.data_dir if args.data_dir else args.images_dir
        ok, _ = verify_split(img_dir, args.labels_dir)

    if not ok:
        raise SystemExit(1)
