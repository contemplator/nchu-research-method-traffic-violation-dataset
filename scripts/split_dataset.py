#!/usr/bin/env python3
"""
scripts/split_dataset.py
YOLO 資料集隨機切分工具
將標註目錄內的圖檔與標籤按 7:2:1 (Train / Val / Test) 切分至標準目錄架構中。
"""

import os
import shutil
import random
import argparse

def split_yolo_dataset(input_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2, seed: int = 42):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"找不到輸入標註目錄: {input_dir}")
        
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)]
    
    if len(images) == 0:
        print(f"⚠️ 在 {input_dir} 內未找到任何圖片，無法進行切分。")
        return
        
    random.seed(seed)
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    print(f"📦 開始切分資料集 (總數: {total} 張):")
    print(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}\n")
    
    for split, files in splits.items():
        img_out = os.path.join(output_dir, split, 'images')
        lbl_out = os.path.join(output_dir, split, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        for f in files:
            base_name = os.path.splitext(f)[0]
            # 複製圖片
            shutil.copy2(os.path.join(input_dir, f), os.path.join(img_out, f))
            # 複製對應標註 txt
            txt_name = base_name + '.txt'
            src_txt = os.path.join(input_dir, txt_name)
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, os.path.join(lbl_out, txt_name))
            else:
                print(f"⚠️ 警告: 圖片 {f} 缺少對應標籤 {txt_name}")
                
    print(f"✅ 資料集切分完畢！成果已儲存至: {output_dir}")
    print(f"   結構如下:")
    print(f"   ├── {output_dir}/train/ (images, labels)")
    print(f"   ├── {output_dir}/val/   (images, labels)")
    print(f"   └── {output_dir}/test/  (images, labels)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 資料集隨機切分工具 (7:2:1)")
    parser.add_argument("--input", type=str, default="data/annotated/", help="已完成標註之輸入資料夾")
    parser.add_argument("--output", type=str, default="data/dataset/", help="切分後輸出資料集路徑")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="訓練集比例 (預設: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="驗證集比例 (預設: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="隨機亂數種子 (預設: 42)")
    
    args = parser.parse_args()
    split_yolo_dataset(args.input, args.output, args.train_ratio, args.val_ratio, args.seed)
