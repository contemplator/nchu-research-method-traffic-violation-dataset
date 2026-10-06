#!/usr/bin/env python3
"""
scripts/extract_frames.py
台灣行車記錄器/交通影片定頻抽幀工具
用於從 MP4/AVI 影片中按秒數間隔自動截取高解析度影格以利標註。
"""

import cv2
import os
import argparse

def extract_frames(video_path: str, output_dir: str, interval_sec: float = 2.0, prefix: str = "frame"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到影片檔案: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval_sec))
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    print(f"🎬 開始處理影片: {video_path}")
    print(f"   總影格數: {total_frames} | FPS: {fps:.2f} | 抽幀間隔: 每 {interval_sec} 秒 (每 {frame_interval} 幀截取一張)")
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            out_filename = f"{video_basename}_{prefix}_{saved_count:04d}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"✅ 抽幀完成！共截取 {saved_count} 張圖片存放於: {output_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="影片自動抽幀工具")
    parser.add_argument("--video", type=str, required=True, help="輸入影片檔案路徑 (例如: sample_videos/dashcam.mp4)")
    parser.add_argument("--output", type=str, default="data/raw/", help="輸出圖片儲存目錄 (預設: data/raw/)")
    parser.add_argument("--interval", type=float, default=2.0, help="抽幀秒數間隔 (預設: 2.0 秒)")
    parser.add_argument("--prefix", type=str, default="frame", help="輸出檔名前綴")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval, args.prefix)
