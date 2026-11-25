import { MAX_VIDEO_DURATION_SEC, MODELS } from '../config/models';
import type { Detection, ModelId, VideoAnalysisResult } from '../types';
import { runModelInference } from './yolo';

export interface VideoProgress {
  currentFrame: number;
  totalFrames: number;
  currentModelId: ModelId;
  timestamp: number;
  elapsedMs: number;
}

function seekVideo(video: HTMLVideoElement, timeSec: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const target = Math.max(0, Math.min(timeSec, Math.max(0, video.duration - 0.01)));
    if (Math.abs(video.currentTime - target) < 0.005 && video.readyState >= 2) {
      resolve();
      return;
    }

    let settled = false;
    const cleanup = () => {
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      window.clearTimeout(timer);
    };

    const onSeeked = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };

    const onError = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error('影片跳轉影格失敗'));
    };

    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    }, 3000);

    video.addEventListener('seeked', onSeeked);
    video.addEventListener('error', onError);
    video.currentTime = target;
  });
}

export function findNearestFrameIndex(timestamps: number[], currentTime: number): number {
  if (timestamps.length === 0) return 0;
  let lo = 0;
  let hi = timestamps.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (timestamps[mid] < currentTime) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo > 0 && Math.abs(timestamps[lo - 1] - currentTime) < Math.abs(timestamps[lo] - currentTime)) {
    return lo - 1;
  }
  return lo;
}

export async function analyzeVideoAllModels(
  video: HTMLVideoElement,
  sampleFps: number,
  onProgress: (progress: VideoProgress) => void,
  signal?: AbortSignal,
): Promise<VideoAnalysisResult> {
  const duration = video.duration;
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error('無法讀取影片長度，請確認檔案為有效的 MP4 格式。');
  }
  if (duration > MAX_VIDEO_DURATION_SEC + 0.5) {
    throw new Error(`影片長度為 ${duration.toFixed(1)} 秒，超過 ${MAX_VIDEO_DURATION_SEC} 秒上限，請剪輯後再上傳。`);
  }

  video.pause();

  const step = 1 / sampleFps;
  const timestamps: number[] = [];
  for (let t = 0; t < duration; t += step) {
    timestamps.push(Math.min(t, Math.max(0, duration - 0.02)));
  }
  if (timestamps.length === 0) timestamps.push(0);

  const frameCanvas = document.createElement('canvas');
  frameCanvas.width = video.videoWidth;
  frameCanvas.height = video.videoHeight;
  const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true })!;

  const perModelFrames: Record<ModelId, Detection[][]> = {
    exp5: [],
    exp7: [],
    exp12: [],
  };
  const perModelInfMs: Record<ModelId, number> = { exp5: 0, exp7: 0, exp12: 0 };
  const perModelTotalMs: Record<ModelId, number> = { exp5: 0, exp7: 0, exp12: 0 };

  const startAll = performance.now();

  for (let i = 0; i < timestamps.length; i++) {
    if (signal?.aborted) {
      throw new DOMException('已取消影片分析', 'AbortError');
    }

    const t = timestamps[i];
    await seekVideo(video, t);
    frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);

    for (const model of MODELS) {
      if (signal?.aborted) {
        throw new DOMException('已取消影片分析', 'AbortError');
      }

      onProgress({
        currentFrame: i + 1,
        totalFrames: timestamps.length,
        currentModelId: model.id,
        timestamp: t,
        elapsedMs: performance.now() - startAll,
      });

      const res = await runModelInference(model.id, frameCanvas);
      perModelFrames[model.id].push(res.detections);
      perModelInfMs[model.id] += res.metrics.inferenceMs;
      perModelTotalMs[model.id] += res.metrics.totalMs;
    }

    // Yield to main thread so progress UI stays responsive
    await new Promise((r) => window.setTimeout(r, 0));
  }

  const n = timestamps.length;
  const buildModelSummary = (id: ModelId) => {
    const avgInferenceMs = perModelInfMs[id] / n;
    const avgTotalMs = perModelTotalMs[id] / n;
    return {
      frames: perModelFrames[id],
      totalMs: perModelTotalMs[id],
      avgInferenceMs,
      avgTotalMs,
      effectiveFps: avgTotalMs > 0 ? 1000 / avgTotalMs : 0,
    };
  };

  return {
    sampleFps,
    duration,
    width: video.videoWidth,
    height: video.videoHeight,
    timestamps,
    perModel: {
      exp5: buildModelSummary('exp5'),
      exp7: buildModelSummary('exp7'),
      exp12: buildModelSummary('exp12'),
    },
  };
}
