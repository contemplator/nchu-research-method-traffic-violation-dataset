import { Cpu, Gauge, Zap } from 'lucide-react';
import { CLASSES, MODELS } from '../config/models';
import { filterDetections } from '../engine/draw';
import type {
  Detection,
  ImageModelResult,
  ModelId,
  ModelLoadState,
  VideoAnalysisResult,
} from '../types';

interface StatsTableProps {
  mode: 'image' | 'video';
  loadStates: Record<ModelId, ModelLoadState>;
  confThresholds: Record<ModelId, number>;
  enabledClasses: Record<number, boolean>;
  imageResults: Partial<Record<ModelId, ImageModelResult>>;
  videoResult: VideoAnalysisResult | null;
  currentFrameIdx: number;
}

export function StatsTable({
  mode,
  loadStates,
  confThresholds,
  enabledClasses,
  imageResults,
  videoResult,
  currentFrameIdx,
}: StatsTableProps) {
  const getVisibleDetections = (id: ModelId): Detection[] => {
    const raw =
      mode === 'image'
        ? imageResults[id]?.detections ?? []
        : videoResult?.perModel[id].frames[currentFrameIdx] ?? [];
    return filterDetections(raw, confThresholds[id], enabledClasses);
  };

  const hasData =
    mode === 'image'
      ? Object.keys(imageResults).length > 0
      : videoResult !== null;

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Gauge className="h-5 w-5 text-sky-400" />
          <h2 className="text-base font-semibold text-slate-100">
            三模型推論速度與偵測數量對照表
          </h2>
        </div>
        <span className="text-xs text-slate-400">
          {mode === 'video' && videoResult
            ? `目前顯示第 ${currentFrameIdx + 1} / ${videoResult.timestamps.length} 抽樣幀（@${videoResult.sampleFps} FPS）`
            : '隨信心門檻與類別開關即時更新'}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-left text-xs sm:text-sm">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="py-2.5 pr-3 font-medium">實驗模型</th>
              <th className="px-2 py-2.5 font-medium">硬體後端</th>
              <th className="px-2 py-2.5 font-medium text-right">推論耗時</th>
              <th className="px-2 py-2.5 font-medium text-right">等效 FPS</th>
              {mode === 'video' && (
                <th className="px-2 py-2.5 font-medium text-right">整段總耗時</th>
              )}
              <th className="px-2 py-2.5 font-medium text-right">總框數</th>
              {CLASSES.map((cls) => (
                <th
                  key={cls.id}
                  className="px-1.5 py-2.5 text-center font-medium"
                  title={cls.name}
                >
                  <span className="inline-flex items-center gap-1">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: cls.color }}
                    />
                    <span className="whitespace-nowrap">{cls.zhName}</span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {MODELS.map((model) => {
              const load = loadStates[model.id];
              const dets = getVisibleDetections(model.id);
              const counts = CLASSES.map(
                (cls) => dets.filter((d) => d.classId === cls.id).length,
              );

              let msText = '—';
              let fpsText = '—';
              let totalText = '—';

              if (mode === 'image' && imageResults[model.id]) {
                const m = imageResults[model.id]!.metrics;
                msText = `${m.totalMs.toFixed(1)} ms`;
                fpsText = `${(1000 / Math.max(m.totalMs, 0.1)).toFixed(1)}`;
              } else if (mode === 'video' && videoResult) {
                const vm = videoResult.perModel[model.id];
                msText = `${vm.avgTotalMs.toFixed(1)} ms/幀`;
                fpsText = `${vm.effectiveFps.toFixed(1)}`;
                totalText = `${(vm.totalMs / 1000).toFixed(2)} s`;
              }

              return (
                <tr key={model.id} className="hover:bg-slate-800/40">
                  <td className="py-3 pr-3">
                    <div className="font-semibold text-slate-100">{model.shortLabel}</div>
                    <div className="text-[11px] text-slate-400">
                      {model.arch} · {model.imgSize}px · 門檻 {confThresholds[model.id].toFixed(2)}
                    </div>
                  </td>
                  <td className="px-2 py-3">
                    {load.backend === 'webgpu' ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
                        <Zap className="h-3 w-3" /> WebGPU
                      </span>
                    ) : load.backend === 'wasm' ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-300">
                        <Cpu className="h-3 w-3" /> WASM CPU
                      </span>
                    ) : (
                      <span className="text-xs text-slate-500">載入中…</span>
                    )}
                  </td>
                  <td className="px-2 py-3 text-right font-mono text-sky-300">
                    {hasData ? msText : '—'}
                  </td>
                  <td className="px-2 py-3 text-right font-mono text-slate-200">
                    {hasData ? fpsText : '—'}
                  </td>
                  {mode === 'video' && (
                    <td className="px-2 py-3 text-right font-mono text-amber-300">
                      {hasData ? totalText : '—'}
                    </td>
                  )}
                  <td className="px-2 py-3 text-right font-mono font-semibold text-white">
                    {hasData ? dets.length : '—'}
                  </td>
                  {counts.map((c, idx) => (
                    <td
                      key={CLASSES[idx].id}
                      className={`px-1.5 py-3 text-center font-mono ${
                        c > 0 ? 'font-semibold text-slate-100' : 'text-slate-600'
                      }`}
                    >
                      {hasData ? c : '—'}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
