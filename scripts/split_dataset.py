#!/usr/bin/env python3
"""
scripts/split_dataset.py
YOLO 資料集隨機切分與自動打包工具
支援圖片（data/annotated/）與標籤（data/labels/）自動配對合併，
按 7:2:1 (Train / Val / Test) 切分至 data/dataset/，並自動生成 data/dataset.zip 供 Colab 雲端訓練。
"""

import os
import shutil
import random
import argparse
import zipfile

def resolve_labels_dir(images_dir: str, labels_dir: str = None) -> str:
    """自動判定標籤存放目錄 (優先 labels_dir -> 預設 data/labels/ -> fallback 到 images_dir)"""
    if labels_dir and os.path.exists(labels_dir):
        return labels_dir
    default_labels = os.path.join(os.path.dirname(images_dir.rstrip('/')), 'labels')
    if os.path.exists(default_labels):
        return default_labels
    return images_dir

def create_dataset_zip(source_dir: str, zip_path: str):
    """跨平台純 Python 打包 zip 檔 (保證 macOS/Linux/Windows 完全相容)"""
    print(f"📦 正在打包資料集壓縮檔: {zip_path} ...")
    root_folder_name = os.path.basename(source_dir.rstrip('/'))  # 'dataset'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in sorted(files):
                if file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.join(root_folder_name, os.path.relpath(file_path, source_dir))
                zipf.write(file_path, rel_path)
    
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ {zip_path} 打包完成！(大小: {zip_size_mb:.2f} MB)")

def split_yolo_dataset(
    images_dir: str = "data/annotated/",
    labels_dir: str = None,
    output_dir: str = "data/dataset/",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    make_zip: bool = True,
    zip_path: str = "data/dataset.zip"
):
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"找不到圖片目錄: {images_dir}")

    actual_labels_dir = resolve_labels_dir(images_dir, labels_dir)
    if not os.path.exists(actual_labels_dir):
        raise FileNotFoundError(f"找不到標籤目錄: {actual_labels_dir}")
        
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
    
    if len(images) == 0:
        print(f"⚠️ 在 {images_dir} 內未找到任何圖片，無法進行切分。")
        return
        
    # 配對驗證：確認標籤存在
    valid_pairs = []
    missing_labels = []
    for img in sorted(images):
        base_name = os.path.splitext(img)[0]
        txt_name = base_name + ".txt"
        txt_path = os.path.join(actual_labels_dir, txt_name)
        if os.path.exists(txt_path):
            valid_pairs.append((img, txt_name))
        else:
            missing_labels.append(img)
            
    if missing_labels:
        print(f"⚠️ 警告: 發現 {len(missing_labels)} 張圖片缺少對應標籤檔，切分時將自動略過：")
        for m in missing_labels[:5]:
            print(f"   • {m}")
        if len(missing_labels) > 5:
            print(f"   ... 其餘 {len(missing_labels) - 5} 張已省略")

    if not valid_pairs:
        print("❌ 沒有任何成對的 (圖片 + 標籤) 可供切分，程序終止。")
        return

    random.seed(seed)
    random.shuffle(valid_pairs)
    
    total = len(valid_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    print(f"🔄 開始合併與切分資料集:")
    print(f"   來源圖片: {images_dir}")
    print(f"   來源標籤: {actual_labels_dir}")
    print(f"   有效樣本總數: {total} 組")
    print(f"   切分比例: Train={len(splits['train'])} | Val={len(splits['val'])} | Test={len(splits['test'])}\n")
    
    # 清理舊的輸出目錄以避免混淆
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    for split_name, pairs in splits.items():
        img_out = os.path.join(output_dir, split_name, 'images')
        lbl_out = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        for img_name, txt_name in pairs:
            # 複製圖片
            shutil.copy2(os.path.join(images_dir, img_name), os.path.join(img_out, img_name))
            # 複製標籤
            shutil.copy2(os.path.join(actual_labels_dir, txt_name), os.path.join(lbl_out, txt_name))
                
    print(f"✅ 資料集標準 YOLO 格式切分完畢！成果已儲存至: {output_dir}")
    print(f"   ├── {output_dir}train/ (images: {len(splits['train'])}, labels: {len(splits['train'])})")
    print(f"   ├── {output_dir}val/   (images: {len(splits['val'])}, labels: {len(splits['val'])})")
    print(f"   └── {output_dir}test/  (images: {len(splits['test'])}, labels: {len(splits['test'])})\n")

    # 自動打包為 dataset.zip
    if make_zip:
        create_dataset_zip(output_dir, zip_path)
        print(f"\n🚀 下一步：直接將 `{zip_path}` 上傳至 Google Drive 根目錄，即可無縫啟動 Colab 消融訓練！\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 資料集隨機切分與打包工具 (7:2:1)")
    parser.add_argument("--images_dir", type=str, default="data/annotated/", help="已完成標註之圖片目錄 (預設: data/annotated/)")
    parser.add_argument("--labels_dir", type=str, default=None, help="YOLO 標籤目錄 (預設: 自動偵測 data/labels/ 或與圖片同目錄)")
    parser.add_argument("--input", type=str, default=None, help="[舊版相容] 同時包含圖檔與標籤之目錄")
    parser.add_argument("--output", type=str, default="data/dataset/", help="切分後輸出資料集路徑 (預設: data/dataset/)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="訓練集比例 (預設: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="驗證集比例 (預設: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="隨機亂數種子 (預設: 42)")
    parser.add_argument("--no_zip", action="store_true", help="不自動打包 dataset.zip")
    parser.add_argument("--zip_path", type=str, default="data/dataset.zip", help="輸出 zip 路徑 (預設: data/dataset.zip)")
    
    args = parser.parse_args()
    img_dir = args.input if args.input else args.images_dir
    lbl_dir = args.labels_dir
    
    split_yolo_dataset(
        images_dir=img_dir,
        labels_dir=lbl_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        make_zip=not args.no_zip,
        zip_path=args.zip_path
    )
