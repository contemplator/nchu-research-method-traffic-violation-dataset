export type ModelId = 'exp5' | 'exp7' | 'exp12';

export type BackendType = 'webgpu' | 'wasm';

export interface ClassDef {
  id: number;
  name: string;
  zhName: string;
  color: string;
}

export interface ModelSpec {
  id: ModelId;
  label: string;
  shortLabel: string;
  arch: string;
  imgSize: number;
  useClahe: boolean;
  defaultConf: number;
  modelPath: string;
  badge: string;
  metrics: {
    map50: number;
    map5095: number;
    f1: number;
  };
}

export interface Detection {
  /** Normalized coordinates [0, 1] relative to original source image/video */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
}

export interface ModelRunMetrics {
  preprocessMs: number;
  inferenceMs: number;
  totalMs: number;
}

export interface ImageModelResult {
  detections: Detection[];
  metrics: ModelRunMetrics;
}

export interface VideoModelResult {
  frames: Detection[][];
  totalMs: number;
  avgInferenceMs: number;
  avgTotalMs: number;
  effectiveFps: number;
}

export interface VideoAnalysisResult {
  sampleFps: number;
  duration: number;
  width: number;
  height: number;
  timestamps: number[];
  perModel: Record<ModelId, VideoModelResult>;
}

export interface ModelLoadState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  backend?: BackendType;
  loadMs?: number;
  error?: string;
}

export interface SampleImage {
  id: string;
  title: string;
  subtitle: string;
  url: string;
}
