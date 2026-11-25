import * as ort from 'onnxruntime-web';
import { CACHE_MIN_CONF, MODEL_MAP, NMS_IOU_THRESHOLD } from '../config/models';
import type { BackendType, Detection, ImageModelResult, ModelId } from '../types';
import { applyLabClaheInPlace } from './clahe';

// Point ONNX Runtime Web to locally bundled WASM assets resolved from the HTML page URL
if (typeof window !== 'undefined') {
  ort.env.wasm.wasmPaths = new URL('./ort/', window.location.href).href;
  ort.env.wasm.numThreads = window.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 4)
    : 1;
}

interface LoadedSession {
  session: ort.InferenceSession;
  backend: BackendType;
  inputName: string;
  outputName: string;
}

const sessionCache = new Map<ModelId, LoadedSession>();

export function hasWebGpuSupport(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export async function loadModelSession(
  modelId: ModelId,
): Promise<{ backend: BackendType; loadMs: number }> {
  const existing = sessionCache.get(modelId);
  if (existing) {
    return { backend: existing.backend, loadMs: 0 };
  }

  const spec = MODEL_MAP[modelId];
  const t0 = performance.now();

  let session: ort.InferenceSession | null = null;
  let backend: BackendType = 'wasm';

  if (hasWebGpuSupport()) {
    try {
      session = await ort.InferenceSession.create(spec.modelPath, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      backend = 'webgpu';
    } catch (err) {
      console.warn(`[${modelId}] WebGPU init failed, falling back to WASM:`, err);
      session = null;
    }
  }

  if (!session) {
    session = await ort.InferenceSession.create(spec.modelPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    backend = 'wasm';
  }

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  sessionCache.set(modelId, { session, backend, inputName, outputName });

  return { backend, loadMs: performance.now() - t0 };
}

interface LetterboxMeta {
  srcW: number;
  srcH: number;
  scale: number;
  padX: number;
  padY: number;
}

// Reusable offscreen canvases keyed by imgSize to avoid GC churn during video pre-analysis
const canvasPool = new Map<number, { lbCanvas: HTMLCanvasElement; contentCanvas: HTMLCanvasElement }>();

function getCanvasPair(imgSize: number) {
  let pair = canvasPool.get(imgSize);
  if (!pair) {
    const lbCanvas = document.createElement('canvas');
    lbCanvas.width = imgSize;
    lbCanvas.height = imgSize;
    const contentCanvas = document.createElement('canvas');
    pair = { lbCanvas, contentCanvas };
    canvasPool.set(imgSize, pair);
  }
  return pair;
}

function preprocessSource(
  source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  imgSize: number,
  useClahe: boolean,
): { tensor: ort.Tensor; meta: LetterboxMeta } {
  const srcW =
    source instanceof HTMLVideoElement
      ? source.videoWidth
      : source instanceof HTMLImageElement
        ? source.naturalWidth
        : source.width;
  const srcH =
    source instanceof HTMLVideoElement
      ? source.videoHeight
      : source instanceof HTMLImageElement
        ? source.naturalHeight
        : source.height;

  const scale = Math.min(imgSize / srcW, imgSize / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.floor((imgSize - newW) / 2);
  const padY = Math.floor((imgSize - newH) / 2);

  const { lbCanvas, contentCanvas } = getCanvasPair(imgSize);
  const lbCtx = lbCanvas.getContext('2d', { willReadFrequently: true })!;

  // Fill YOLO standard letterbox gray (114, 114, 114)
  lbCtx.fillStyle = '#727272';
  lbCtx.fillRect(0, 0, imgSize, imgSize);

  if (useClahe) {
    // Apply LAB-CLAHE only on the actual image region before placing on letterbox canvas
    if (contentCanvas.width !== newW || contentCanvas.height !== newH) {
      contentCanvas.width = newW;
      contentCanvas.height = newH;
    }
    const cCtx = contentCanvas.getContext('2d', { willReadFrequently: true })!;
    cCtx.drawImage(source, 0, 0, newW, newH);
    const contentData = cCtx.getImageData(0, 0, newW, newH);
    applyLabClaheInPlace(contentData, 2.0, 8);
    cCtx.putImageData(contentData, 0, 0);
    lbCtx.drawImage(contentCanvas, padX, padY);
  } else {
    lbCtx.drawImage(source, padX, padY, newW, newH);
  }

  const rgba = lbCtx.getImageData(0, 0, imgSize, imgSize).data;
  const area = imgSize * imgSize;
  const floatData = new Float32Array(3 * area);

  // HWC RGBA -> CHW RGB normalized [0, 1]
  for (let i = 0, p = 0; i < area; i++, p += 4) {
    floatData[i] = rgba[p] * (1 / 255);
    floatData[area + i] = rgba[p + 1] * (1 / 255);
    floatData[2 * area + i] = rgba[p + 2] * (1 / 255);
  }

  const tensor = new ort.Tensor('float32', floatData, [1, 3, imgSize, imgSize]);
  return { tensor, meta: { srcW, srcH, scale, padX, padY } };
}

function iou(a: Detection, b: Detection): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function postprocessYolo(
  data: Float32Array,
  dims: readonly number[],
  meta: LetterboxMeta,
  minConf = CACHE_MIN_CONF,
  iouThreshold = NMS_IOU_THRESHOLD,
): Detection[] {
  // Expected output shape from YOLOv7 decoded export: [1, numBoxes, 5 + numClasses]
  const numBoxes = dims[1];
  const stride = dims[2];
  const numClasses = stride - 5;

  const candidates: Detection[] = [];
  const { srcW, srcH, scale, padX, padY } = meta;

  for (let i = 0; i < numBoxes; i++) {
    const offset = i * stride;
    const objConf = data[offset + 4];
    if (objConf < minConf) continue;

    let bestClass = 0;
    let bestProb = data[offset + 5];
    for (let c = 1; c < numClasses; c++) {
      const p = data[offset + 5 + c];
      if (p > bestProb) {
        bestProb = p;
        bestClass = c;
      }
    }

    const score = objConf * bestProb;
    if (score < minConf) continue;

    const cx = data[offset];
    const cy = data[offset + 1];
    const w = data[offset + 2];
    const h = data[offset + 3];

    const x1 = Math.max(0, Math.min(1, (cx - w / 2 - padX) / scale / srcW));
    const y1 = Math.max(0, Math.min(1, (cy - h / 2 - padY) / scale / srcH));
    const x2 = Math.max(0, Math.min(1, (cx + w / 2 - padX) / scale / srcW));
    const y2 = Math.max(0, Math.min(1, (cy + h / 2 - padY) / scale / srcH));

    if (x2 <= x1 || y2 <= y1) continue;

    candidates.push({ x1, y1, x2, y2, score, classId: bestClass });
  }

  // Class-aware NMS
  candidates.sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const det of candidates) {
    let suppressed = false;
    for (const prev of kept) {
      if (prev.classId === det.classId && iou(prev, det) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push(det);
  }

  return kept;
}

export async function runModelInference(
  modelId: ModelId,
  source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
): Promise<ImageModelResult> {
  const loaded = sessionCache.get(modelId);
  if (!loaded) {
    throw new Error(`模型 ${modelId} 尚未載入`);
  }

  const spec = MODEL_MAP[modelId];
  const tPre0 = performance.now();
  const { tensor, meta } = preprocessSource(source, spec.imgSize, spec.useClahe);
  const preprocessMs = performance.now() - tPre0;

  const tInf0 = performance.now();
  const feeds: Record<string, ort.Tensor> = { [loaded.inputName]: tensor };
  const results = await loaded.session.run(feeds);
  const outputTensor = results[loaded.outputName];
  const inferenceMs = performance.now() - tInf0;

  const outputData = outputTensor.data as Float32Array;
  const detections = postprocessYolo(outputData, outputTensor.dims, meta);

  // Release tensor memory promptly (important for WebGPU / video loops)
  tensor.dispose();
  outputTensor.dispose();

  return {
    detections,
    metrics: {
      preprocessMs,
      inferenceMs,
      totalMs: preprocessMs + inferenceMs,
    },
  };
}
