import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Film,
  Image as ImageIcon,
  Loader2,
  RotateCcw,
  SlidersHorizontal,
  Upload,
  XCircle,
} from 'lucide-react';
import {
  CLASSES,
  MAX_VIDEO_DURATION_SEC,
  MODEL_MAP,
  MODELS,
  SAMPLE_IMAGES,
} from './config/models';
import { ModelComparisonGrid } from './components/ModelComparisonGrid';
import { StatsTable } from './components/StatsTable';
import { analyzeVideoAllModels, type VideoProgress } from './engine/videoAnalyzer';
import { loadModelSession, runModelInference } from './engine/yolo';
import type {
  ImageModelResult,
  ModelId,
  ModelLoadState,
  VideoAnalysisResult,
} from './types';

const DEFAULT_CONFS: Record<ModelId, number> = {
  exp5: MODEL_MAP.exp5.defaultConf,
  exp7: MODEL_MAP.exp7.defaultConf,
  exp12: MODEL_MAP.exp12.defaultConf,
};

const DEFAULT_ENABLED_CLASSES: Record<number, boolean> = Object.fromEntries(
  CLASSES.map((c) => [c.id, true]),
);

export function App() {
  const [loadStates, setLoadStates] = useState<Record<ModelId, ModelLoadState>>({
    exp5: { status: 'loading' },
    exp7: { status: 'loading' },
    exp12: { status: 'loading' },
  });

  const [mode, setMode] = useState<'image' | 'video'>('image');
  const [confThresholds, setConfThresholds] = useState<Record<ModelId, number>>(DEFAULT_CONFS);
  const [enabledClasses, setEnabledClasses] =
    useState<Record<number, boolean>>(DEFAULT_ENABLED_CLASSES);

  // Image state
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null);
  const [imageResults, setImageResults] = useState<Partial<Record<ModelId, ImageModelResult>>>({});
  const [isInferringImage, setIsInferringImage] = useState(false);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);

  // Video state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [sampleFps, setSampleFps] = useState<number>(5);
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);
  const [videoProgress, setVideoProgress] = useState<VideoProgress | null>(null);
  const [videoResult, setVideoResult] = useState<VideoAnalysisResult | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [loop, setLoop] = useState(true);

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const allModelsReady = MODELS.every((m) => loadStates[m.id].status === 'ready');

  // Load all 3 ONNX models on mount (sequentially to avoid memory spike during WASM init)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (const model of MODELS) {
        if (cancelled) return;
        setLoadStates((prev) => ({ ...prev, [model.id]: { status: 'loading' } }));
        try {
          const { backend, loadMs } = await loadModelSession(model.id);
          if (cancelled) return;
          setLoadStates((prev) => ({
            ...prev,
            [model.id]: { status: 'ready', backend, loadMs },
          }));
        } catch (err) {
          if (cancelled) return;
          const message = err instanceof Error ? err.message : String(err);
          setLoadStates((prev) => ({
            ...prev,
            [model.id]: { status: 'error', error: message },
          }));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Run all 3 models on an HTMLImageElement
  const runImageThroughAllModels = useCallback(async (img: HTMLImageElement) => {
    setErrorMsg(null);
    setMode('image');
    setVideoResult(null);
    setIsPlaying(false);
    setImageElement(img);
    setImageResults({});
    setIsInferringImage(true);

    try {
      const next: Partial<Record<ModelId, ImageModelResult>> = {};
      for (const model of MODELS) {
        const res = await runModelInference(model.id, img);
        next[model.id] = res;
        setImageResults({ ...next });
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : '圖片推論失敗');
    } finally {
      setIsInferringImage(false);
    }
  }, []);

  // Load a sample image when clicked (or automatically once all models are ready)
  const loadSampleImage = useCallback(
    (sampleId: string) => {
      const sample = SAMPLE_IMAGES.find((s) => s.id === sampleId);
      if (!sample) return;
      setActiveSampleId(sample.id);
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        void runImageThroughAllModels(img);
      };
      img.onerror = () => setErrorMsg(`無法載入範例圖片：${sample.title}`);
      img.src = sample.url;
    },
    [runImageThroughAllModels],
  );

  // Auto-run first sample image once all 3 models finish loading
  const didAutoRunRef = useRef(false);
  useEffect(() => {
    if (allModelsReady && !didAutoRunRef.current && !imageElement && !videoUrl) {
      didAutoRunRef.current = true;
      loadSampleImage(SAMPLE_IMAGES[0].id);
    }
  }, [allModelsReady, imageElement, videoUrl, loadSampleImage]);

  // Handle user file upload (Image or <= 30s MP4)
  const handleFileChange = async (file: File | undefined) => {
    if (!file) return;
    setErrorMsg(null);
    setActiveSampleId(null);

    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        void runImageThroughAllModels(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        setErrorMsg('無法讀取這張圖片，請確認格式為 JPG / PNG / WebP。');
      };
      img.src = url;
      return;
    }

    if (file.type.startsWith('video/') || file.name.toLowerCase().endsWith('.mp4')) {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      const url = URL.createObjectURL(file);
      setMode('video');
      setImageElement(null);
      setImageResults({});
      setVideoResult(null);
      setIsPlaying(false);
      setCurrentTime(0);
      setCurrentFrameIdx(0);
      setVideoUrl(url);
      return;
    }

    setErrorMsg('僅支援圖片檔（JPG / PNG / WebP）或 30 秒內的 MP4 影片。');
  };

  // Start pre-analysis when video metadata is loaded
  const startVideoAnalysis = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !allModelsReady) return;

    if (video.duration > MAX_VIDEO_DURATION_SEC + 0.5) {
      setErrorMsg(
        `此影片長度為 ${video.duration.toFixed(1)} 秒，超過 ${MAX_VIDEO_DURATION_SEC} 秒上限，請上傳 30 秒內的 MP4。`,
      );
      return;
    }

    abortControllerRef.current?.abort();
    const ac = new AbortController();
    abortControllerRef.current = ac;

    setErrorMsg(null);
    setIsAnalyzingVideo(true);
    setVideoResult(null);
    setVideoProgress(null);

    try {
      const result = await analyzeVideoAllModels(
        video,
        sampleFps,
        (prog) => setVideoProgress(prog),
        ac.signal,
      );
      setVideoResult(result);
      video.currentTime = 0;
      setCurrentTime(0);
      setCurrentFrameIdx(0);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setErrorMsg('已取消影片預分析。');
      } else {
        setErrorMsg(err instanceof Error ? err.message : '影片分析失敗');
      }
    } finally {
      setIsAnalyzingVideo(false);
    }
  }, [allModelsReady, sampleFps]);

  const handleTogglePlay = () => {
    const video = videoRef.current;
    if (!video || !videoResult) return;
    if (video.paused || video.ended) {
      if (video.ended) video.currentTime = 0;
      video.playbackRate = playbackRate;
      video.loop = loop;
      void video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const handleSeek = (timeSec: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = timeSec;
    setCurrentTime(timeSec);
  };

  const handleChangePlaybackRate = (rate: number) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  };

  const handleToggleLoop = () => {
    setLoop((prev) => {
      const next = !prev;
      if (videoRef.current) videoRef.current.loop = next;
      return next;
    });
  };

  const handleSyncFrameIdx = useCallback((idx: number, timeSec: number) => {
    setCurrentFrameIdx(idx);
    setCurrentTime(timeSec);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Hidden master video element for frame-accurate 3-canvas sync */}
      {videoUrl && (
        <video
          ref={videoRef}
          src={videoUrl}
          playsInline
          muted
          preload="auto"
          className="hidden"
          onLoadedMetadata={() => {
            void startVideoAnalysis();
          }}
          onEnded={() => {
            if (!loop) setIsPlaying(false);
          }}
        />
      )}

      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-md bg-sky-500/20 px-2 py-0.5 text-xs font-semibold text-sky-300">
                NCHU Research Method
              </span>
              <span className="text-xs text-slate-400">
                ONNX Runtime Web · 純瀏覽器端本地推論
              </span>
            </div>
            <h1 className="mt-1 text-lg font-bold tracking-tight text-white sm:text-xl">
              台灣道路交通設施偵測 — YOLOv7 三模型即時對比展示
            </h1>
          </div>

          {/* Model loading status pills */}
          <div className="flex flex-wrap items-center gap-2">
            {MODELS.map((m) => {
              const st = loadStates[m.id];
              return (
                <div
                  key={m.id}
                  className="inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900 px-3 py-1 text-xs"
                >
                  {st.status === 'ready' ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                  ) : st.status === 'error' ? (
                    <XCircle className="h-3.5 w-3.5 text-rose-400" />
                  ) : (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-400" />
                  )}
                  <span className="font-medium text-slate-200">{m.id.toUpperCase()}</span>
                  {st.backend && (
                    <span className="text-[10px] uppercase text-slate-400">{st.backend}</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-5 px-4 py-6 sm:px-6">
        {/* Upload & Sample Control Panel */}
        <section className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          {/* Upload Zone */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                <Upload className="h-4 w-4 text-sky-400" />
                上傳圖片或 30 秒內 MP4 影片
              </h2>
              <div className="flex items-center gap-1.5 text-xs text-slate-400">
                <span>影片抽樣率</span>
                <select
                  value={sampleFps}
                  onChange={(e) => setSampleFps(Number(e.target.value))}
                  disabled={isAnalyzingVideo}
                  className="rounded-md border border-slate-700 bg-slate-800 px-2 py-0.5 text-xs text-slate-200"
                >
                  <option value={3}>3 FPS（最快）</option>
                  <option value={5}>5 FPS（推薦）</option>
                  <option value={10}>10 FPS（細緻）</option>
                </select>
              </div>
            </div>

            <div
              onClick={() => allModelsReady && !isAnalyzingVideo && fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!allModelsReady || isAnalyzingVideo) return;
                void handleFileChange(e.dataTransfer.files[0]);
              }}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-5 text-center transition ${
                allModelsReady && !isAnalyzingVideo
                  ? 'border-slate-700 bg-slate-950/60 hover:border-sky-500/70 hover:bg-slate-900'
                  : 'cursor-not-allowed border-slate-800 bg-slate-950/40 opacity-60'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,video/mp4"
                className="hidden"
                onChange={(e) => {
                  void handleFileChange(e.target.files?.[0]);
                  e.target.value = '';
                }}
              />
              <div className="flex items-center gap-3 text-sky-400">
                <ImageIcon className="h-6 w-6" />
                <Film className="h-6 w-6" />
              </div>
              <p className="mt-2 text-sm font-medium text-slate-200">
                {allModelsReady ? '點擊選擇或拖曳檔案到此處' : '正在載入三個 ONNX 模型，請稍候…'}
              </p>
              <p className="mt-1 text-xs text-slate-400">
                支援 JPG / PNG 圖片（即時出圖）或 ≤ 30 秒 MP4（預分析後同步來回播放）
              </p>
            </div>

            {mode === 'video' && videoUrl && !isAnalyzingVideo && (
              <button
                type="button"
                onClick={() => void startVideoAnalysis()}
                className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-800 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
              >
                以目前抽樣率（{sampleFps} FPS）重新分析此影片
              </button>
            )}
          </div>

          {/* Built-in Test Samples + Class Filters */}
          <div className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-7">
            <div>
              <h2 className="mb-2.5 text-sm font-semibold text-slate-100">
                一鍵試測（內建獨立 Test 集台灣道路影像）
              </h2>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {SAMPLE_IMAGES.map((sample) => {
                  const active = mode === 'image' && activeSampleId === sample.id;
                  return (
                    <button
                      key={sample.id}
                      type="button"
                      disabled={!allModelsReady || isAnalyzingVideo}
                      onClick={() => loadSampleImage(sample.id)}
                      className={`group overflow-hidden rounded-xl border text-left transition ${
                        active
                          ? 'border-sky-400 bg-sky-500/10'
                          : 'border-slate-800 bg-slate-950/60 hover:border-slate-600'
                      } disabled:opacity-50`}
                    >
                      <div className="aspect-video w-full overflow-hidden bg-slate-900">
                        <img
                          src={sample.url}
                          alt={sample.title}
                          className="h-full w-full object-cover transition group-hover:scale-105"
                        />
                      </div>
                      <div className="p-2">
                        <div className="truncate text-xs font-semibold text-slate-200">
                          {sample.title}
                        </div>
                        <div className="truncate text-[10px] text-slate-400">
                          {sample.subtitle}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* 6-Class Filter Switches */}
            <div className="border-t border-slate-800/80 pt-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                  <SlidersHorizontal className="h-3.5 w-3.5 text-sky-400" />
                  6 大交通設施類別顯示開關
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setEnabledClasses(DEFAULT_ENABLED_CLASSES);
                    setConfThresholds(DEFAULT_CONFS);
                  }}
                  className="inline-flex items-center gap-1 text-[11px] text-sky-400 hover:text-sky-300"
                >
                  <RotateCcw className="h-3 w-3" /> 全部重設為黃金設定
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {CLASSES.map((cls) => {
                  const enabled = enabledClasses[cls.id] !== false;
                  return (
                    <button
                      key={cls.id}
                      type="button"
                      onClick={() =>
                        setEnabledClasses((prev) => ({ ...prev, [cls.id]: !enabled }))
                      }
                      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition ${
                        enabled
                          ? 'border-slate-600 bg-slate-800 text-slate-100'
                          : 'border-slate-800 bg-slate-950 text-slate-500 line-through'
                      }`}
                    >
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{
                          backgroundColor: cls.color,
                          opacity: enabled ? 1 : 0.3,
                        }}
                      />
                      {cls.zhName}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        {/* Error Banner */}
        {(errorMsg || MODELS.some((m) => loadStates[m.id].status === 'error')) && (
          <div className="space-y-1 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {errorMsg && (
              <div className="flex items-center gap-2">
                <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
                <span>{errorMsg}</span>
              </div>
            )}
            {MODELS.filter((m) => loadStates[m.id].status === 'error').map((m) => (
              <div key={m.id} className="flex items-center gap-2 text-xs">
                <AlertCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                <span>
                  {m.shortLabel} 載入失敗：{loadStates[m.id].error}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Image Inference Loading Bar */}
        {isInferringImage && (
          <div className="flex items-center gap-2 rounded-xl border border-sky-500/30 bg-sky-500/10 px-4 py-2.5 text-xs text-sky-200">
            <Loader2 className="h-4 w-4 animate-spin text-sky-400" />
            <span>正在以 Exp 5 / Exp 7 / Exp 12 進行瀏覽器端推論…</span>
          </div>
        )}

        {/* Video Pre-Analysis Progress Banner */}
        {isAnalyzingVideo && videoProgress && (
          <div className="rounded-2xl border border-sky-500/40 bg-slate-900/90 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
              <div className="flex items-center gap-2 font-medium text-sky-200">
                <Loader2 className="h-4 w-4 animate-spin text-sky-400" />
                <span>
                  正在預分析影片：第 {videoProgress.currentFrame} / {videoProgress.totalFrames} 幀
                  （目前執行 {videoProgress.currentModelId.toUpperCase()} · 時間點{' '}
                  {videoProgress.timestamp.toFixed(1)}s）
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-slate-400">
                  已耗時 {(videoProgress.elapsedMs / 1000).toFixed(1)}s
                </span>
                <button
                  type="button"
                  onClick={() => abortControllerRef.current?.abort()}
                  className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-2.5 py-1 text-xs text-rose-300 hover:bg-rose-500/20"
                >
                  取消分析
                </button>
              </div>
            </div>
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-sky-400 transition-all duration-150"
                style={{
                  width: `${Math.round(
                    (videoProgress.currentFrame / Math.max(1, videoProgress.totalFrames)) * 100,
                  )}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* 3-Model Side-by-Side Comparison Grid */}
        <ModelComparisonGrid
          mode={mode}
          imageElement={imageElement}
          videoElement={videoRef.current}
          loadStates={loadStates}
          confThresholds={confThresholds}
          onChangeConf={(id, value) =>
            setConfThresholds((prev) => ({ ...prev, [id]: value }))
          }
          onResetConf={(id) =>
            setConfThresholds((prev) => ({ ...prev, [id]: DEFAULT_CONFS[id] }))
          }
          enabledClasses={enabledClasses}
          imageResults={imageResults}
          videoResult={videoResult}
          isPlaying={isPlaying}
          onTogglePlay={handleTogglePlay}
          currentTime={currentTime}
          onSeek={handleSeek}
          playbackRate={playbackRate}
          onChangePlaybackRate={handleChangePlaybackRate}
          loop={loop}
          onToggleLoop={handleToggleLoop}
          currentFrameIdx={currentFrameIdx}
          onSyncFrameIdx={handleSyncFrameIdx}
        />

        {/* 3-Model Speed & Detection Count Comparison Table */}
        <StatsTable
          mode={mode}
          loadStates={loadStates}
          confThresholds={confThresholds}
          enabledClasses={enabledClasses}
          imageResults={imageResults}
          videoResult={videoResult}
          currentFrameIdx={currentFrameIdx}
        />
      </main>
    </div>
  );
}

export default App;
