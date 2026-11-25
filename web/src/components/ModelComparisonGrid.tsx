import { useEffect, useRef } from 'react';
import {
  Cpu,
  Pause,
  Play,
  Repeat,
  RotateCcw,
  SkipBack,
  SkipForward,
  Sparkles,
  Zap,
} from 'lucide-react';
import { MODELS } from '../config/models';
import { drawDetectionsOnCanvas, filterDetections } from '../engine/draw';
import { findNearestFrameIndex } from '../engine/videoAnalyzer';
import type {
  ImageModelResult,
  ModelId,
  ModelLoadState,
  VideoAnalysisResult,
} from '../types';

interface ModelComparisonGridProps {
  mode: 'image' | 'video';
  imageElement: HTMLImageElement | null;
  videoElement: HTMLVideoElement | null;
  loadStates: Record<ModelId, ModelLoadState>;
  confThresholds: Record<ModelId, number>;
  onChangeConf: (id: ModelId, value: number) => void;
  onResetConf: (id: ModelId) => void;
  enabledClasses: Record<number, boolean>;
  imageResults: Partial<Record<ModelId, ImageModelResult>>;
  videoResult: VideoAnalysisResult | null;
  isPlaying: boolean;
  onTogglePlay: () => void;
  currentTime: number;
  onSeek: (timeSec: number) => void;
  playbackRate: number;
  onChangePlaybackRate: (rate: number) => void;
  loop: boolean;
  onToggleLoop: () => void;
  currentFrameIdx: number;
  onSyncFrameIdx: (idx: number, timeSec: number) => void;
}

export function ModelComparisonGrid({
  mode,
  imageElement,
  videoElement,
  loadStates,
  confThresholds,
  onChangeConf,
  onResetConf,
  enabledClasses,
  imageResults,
  videoResult,
  isPlaying,
  onTogglePlay,
  currentTime,
  onSeek,
  playbackRate,
  onChangePlaybackRate,
  loop,
  onToggleLoop,
  currentFrameIdx,
  onSyncFrameIdx,
}: ModelComparisonGridProps) {
  const canvasRefs = useRef<Record<ModelId, HTMLCanvasElement | null>>({
    exp5: null,
    exp7: null,
    exp12: null,
  });

  // Render Image Mode whenever image, detections, thresholds, or class filters change
  useEffect(() => {
    if (mode !== 'image' || !imageElement) return;
    for (const model of MODELS) {
      const canvas = canvasRefs.current[model.id];
      if (!canvas) continue;
      if (
        canvas.width !== imageElement.naturalWidth ||
        canvas.height !== imageElement.naturalHeight
      ) {
        canvas.width = imageElement.naturalWidth;
        canvas.height = imageElement.naturalHeight;
      }
      const dets = imageResults[model.id]?.detections ?? [];
      drawDetectionsOnCanvas(
        canvas,
        imageElement,
        dets,
        confThresholds[model.id],
        enabledClasses,
      );
    }
  }, [mode, imageElement, imageResults, confThresholds, enabledClasses]);

  // Render Video Mode continuously while playing, and immediately when paused/scrubbed/filtered
  useEffect(() => {
    if (mode !== 'video' || !videoElement || !videoResult) return;

    let rafId = 0;

    const renderAllCanvases = () => {
      const t = videoElement.currentTime;
      const idx = findNearestFrameIndex(videoResult.timestamps, t);
      onSyncFrameIdx(idx, t);

      for (const model of MODELS) {
        const canvas = canvasRefs.current[model.id];
        if (!canvas) continue;
        if (canvas.width !== videoResult.width || canvas.height !== videoResult.height) {
          canvas.width = videoResult.width;
          canvas.height = videoResult.height;
        }
        const dets = videoResult.perModel[model.id].frames[idx] ?? [];
        drawDetectionsOnCanvas(
          canvas,
          videoElement,
          dets,
          confThresholds[model.id],
          enabledClasses,
        );
      }

      if (!videoElement.paused && !videoElement.ended) {
        rafId = window.requestAnimationFrame(renderAllCanvases);
      }
    };

    renderAllCanvases();

    const onSeeked = () => renderAllCanvases();
    const onPlay = () => renderAllCanvases();
    videoElement.addEventListener('seeked', onSeeked);
    videoElement.addEventListener('play', onPlay);

    return () => {
      window.cancelAnimationFrame(rafId);
      videoElement.removeEventListener('seeked', onSeeked);
      videoElement.removeEventListener('play', onPlay);
    };
  }, [mode, videoElement, videoResult, isPlaying, confThresholds, enabledClasses, onSyncFrameIdx]);

  const stepFrame = (delta: number) => {
    if (!videoResult) return;
    const nextIdx = Math.max(
      0,
      Math.min(videoResult.timestamps.length - 1, currentFrameIdx + delta),
    );
    onSeek(videoResult.timestamps[nextIdx]);
  };

  return (
    <div className="space-y-4">
      {/* Synchronized Video Timeline Controller */}
      {mode === 'video' && videoResult && (
        <div className="rounded-2xl border border-sky-500/30 bg-slate-900/90 p-4 shadow-lg shadow-sky-950/20">
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={onTogglePlay}
              className="inline-flex items-center gap-1.5 rounded-xl bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-sky-400"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isPlaying ? '暫停' : '同步播放'}
            </button>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => stepFrame(-1)}
                title="上一個抽樣幀"
                className="rounded-lg border border-slate-700 bg-slate-800 p-2 text-slate-200 hover:bg-slate-700"
              >
                <SkipBack className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => stepFrame(1)}
                title="下一個抽樣幀"
                className="rounded-lg border border-slate-700 bg-slate-800 p-2 text-slate-200 hover:bg-slate-700"
              >
                <SkipForward className="h-4 w-4" />
              </button>
            </div>

            <div className="flex min-w-[200px] flex-1 items-center gap-3">
              <span className="w-12 text-right font-mono text-xs text-sky-300">
                {currentTime.toFixed(2)}s
              </span>
              <input
                type="range"
                min={0}
                max={videoResult.duration}
                step={0.01}
                value={Math.min(currentTime, videoResult.duration)}
                onChange={(e) => onSeek(Number(e.target.value))}
                className="h-2 flex-1 cursor-pointer rounded-lg bg-slate-800"
              />
              <span className="w-12 font-mono text-xs text-slate-400">
                {videoResult.duration.toFixed(2)}s
              </span>
            </div>

            <div className="flex items-center gap-2">
              <select
                value={playbackRate}
                onChange={(e) => onChangePlaybackRate(Number(e.target.value))}
                className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-slate-200"
              >
                <option value={0.25}>0.25x 慢速</option>
                <option value={0.5}>0.5x 慢速</option>
                <option value={1}>1.0x 原速</option>
                <option value={1.5}>1.5x 倍速</option>
              </select>

              <button
                type="button"
                onClick={onToggleLoop}
                className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${
                  loop
                    ? 'border-sky-500/60 bg-sky-500/15 text-sky-300'
                    : 'border-slate-700 bg-slate-800 text-slate-400'
                }`}
              >
                <Repeat className="h-3.5 w-3.5" />
                循環
              </button>
            </div>
          </div>
          <p className="mt-2 text-xs text-slate-400">
            三格畫面已與快取結果同步，可自由來回拖拉時間軸或微調下方門檻滑桿，無需重新推論。
          </p>
        </div>
      )}

      {/* 3-Column Side-by-Side Model Comparison Cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {MODELS.map((model) => {
          const load = loadStates[model.id];
          const conf = confThresholds[model.id];
          const rawDets =
            mode === 'image'
              ? imageResults[model.id]?.detections ?? []
              : videoResult?.perModel[model.id].frames[currentFrameIdx] ?? [];
          const visibleCount = filterDetections(rawDets, conf, enabledClasses).length;
          const isExp12 = model.id === 'exp12';

          let speedBadge = '待推論';
          if (mode === 'image' && imageResults[model.id]) {
            const m = imageResults[model.id]!.metrics;
            speedBadge = `${m.totalMs.toFixed(1)} ms`;
          } else if (mode === 'video' && videoResult) {
            const vm = videoResult.perModel[model.id];
            speedBadge = `均 ${vm.avgTotalMs.toFixed(0)} ms · ${vm.effectiveFps.toFixed(1)} FPS`;
          }

          return (
            <div
              key={model.id}
              className={`flex flex-col overflow-hidden rounded-2xl border bg-slate-900/80 ${
                isExp12 ? 'border-sky-500/50 ring-1 ring-sky-500/20' : 'border-slate-800'
              }`}
            >
              {/* Card Header */}
              <div className="border-b border-slate-800/80 p-3.5">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-1.5">
                      <h3 className="text-sm font-bold text-slate-100">{model.shortLabel}</h3>
                      {isExp12 && <Sparkles className="h-3.5 w-3.5 text-sky-400" />}
                    </div>
                    <p className="mt-0.5 text-[11px] text-slate-400">
                      {model.arch} · {model.imgSize}px · Test mAP@0.5 {model.metrics.map50}%
                    </p>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                      isExp12
                        ? 'bg-sky-500/20 text-sky-300'
                        : 'bg-slate-800 text-slate-300'
                    }`}
                  >
                    {model.badge}
                  </span>
                </div>

                <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div className="flex items-center gap-1.5">
                    {load.backend === 'webgpu' ? (
                      <span className="inline-flex items-center gap-1 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[11px] text-emerald-300">
                        <Zap className="h-3 w-3" /> WebGPU
                      </span>
                    ) : load.backend === 'wasm' ? (
                      <span className="inline-flex items-center gap-1 rounded bg-amber-500/15 px-1.5 py-0.5 text-[11px] text-amber-300">
                        <Cpu className="h-3 w-3" /> WASM
                      </span>
                    ) : (
                      <span className="text-[11px] text-slate-500">
                        {load.status === 'error' ? '載入失敗' : '載入模型中…'}
                      </span>
                    )}
                    <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] text-sky-300">
                      {speedBadge}
                    </span>
                  </div>
                  <span className="font-mono text-xs font-semibold text-slate-200">
                    偵測：{visibleCount} 個
                  </span>
                </div>
              </div>

              {/* Canvas Viewport */}
              <div className="relative flex aspect-video items-center justify-center bg-slate-950">
                <canvas
                  ref={(el) => {
                    canvasRefs.current[model.id] = el;
                  }}
                  className="max-h-full max-w-full object-contain"
                />
                {!imageElement && !videoResult && (
                  <span className="absolute text-xs text-slate-500">
                    請上傳圖片／影片或點選範例圖片
                  </span>
                )}
              </div>

              {/* Per-Model Confidence Slider */}
              <div className="border-t border-slate-800/80 bg-slate-900/60 p-3">
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="text-slate-400">
                    信心門檻{' '}
                    <span className="font-mono font-semibold text-sky-300">
                      {conf.toFixed(2)}
                    </span>
                    <span className="ml-1.5 text-[11px] text-slate-500">
                      (黃金值 {model.defaultConf.toFixed(2)})
                    </span>
                  </span>
                  {Math.abs(conf - model.defaultConf) > 0.001 && (
                    <button
                      type="button"
                      onClick={() => onResetConf(model.id)}
                      className="inline-flex items-center gap-1 text-[11px] text-sky-400 hover:text-sky-300"
                    >
                      <RotateCcw className="h-3 w-3" /> 重設
                    </button>
                  )}
                </div>
                <input
                  type="range"
                  min={0.15}
                  max={0.9}
                  step={0.05}
                  value={conf}
                  onChange={(e) => onChangeConf(model.id, Number(e.target.value))}
                  className="h-1.5 w-full cursor-pointer rounded-lg bg-slate-800"
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
