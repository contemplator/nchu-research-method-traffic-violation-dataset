import type { ClassDef, ModelId, ModelSpec, SampleImage } from '../types';

export const CLASSES: ClassDef[] = [
  { id: 0, name: 'traffic_light', zhName: '交通號誌', color: '#ef4444' },
  { id: 1, name: 'crosswalk', zhName: '行人穿越道', color: '#38bdf8' },
  { id: 2, name: 'two_stage_left_turn_box', zhName: '機車待轉區', color: '#c084fc' },
  { id: 3, name: 'Sign_follow', zhName: '遵行標誌', color: '#22c55e' },
  { id: 4, name: 'Sign_prohibit', zhName: '禁制標誌', color: '#fb923c' },
  { id: 5, name: 'Sign_limit', zhName: '速限標誌', color: '#facc15' },
];

export const MODELS: ModelSpec[] = [
  {
    id: 'exp5',
    label: 'Exp 5：Tiny-640 (v1 調參)',
    shortLabel: 'Exp 5 · Tiny-640',
    arch: 'YOLOv7-tiny',
    imgSize: 640,
    useClahe: false,
    defaultConf: 0.45,
    modelPath: './models/exp5_tiny_640.onnx',
    badge: '最輕量高速',
    metrics: { map50: 43.1, map5095: 24.2, f1: 52.3 },
  },
  {
    id: 'exp7',
    label: 'Exp 7：Base-640 (100ep 預設)',
    shortLabel: 'Exp 7 · Base-640',
    arch: 'YOLOv7 Base',
    imgSize: 640,
    useClahe: false,
    defaultConf: 0.35,
    modelPath: './models/exp7_base_640.onnx',
    badge: '大骨幹對照',
    metrics: { map50: 43.3, map5095: 23.9, f1: 56.3 },
  },
  {
    id: 'exp12',
    label: 'Exp 12：Tiny-960 + LAB-CLAHE',
    shortLabel: 'Exp 12 · Tiny-960 CLAHE',
    arch: 'YOLOv7-tiny + CLAHE',
    imgSize: 960,
    useClahe: true,
    defaultConf: 0.60,
    modelPath: './models/exp12_tiny_960_clahe.onnx',
    badge: 'Test mAP 冠軍',
    metrics: { map50: 60.9, map5095: 35.2, f1: 63.8 },
  },
];

export const MODEL_MAP: Record<ModelId, ModelSpec> = {
  exp5: MODELS[0],
  exp7: MODELS[1],
  exp12: MODELS[2],
};

export const SAMPLE_IMAGES: SampleImage[] = [
  {
    id: 'intersection-1',
    title: '市區多設施路口 A',
    subtitle: '號誌・斑馬線・待轉區・遵行／禁制標誌',
    url: './samples/sample_intersection_1.jpg',
  },
  {
    id: 'intersection-2',
    title: '市區多設施路口 B',
    subtitle: '複合路口標線與多組紅綠燈',
    url: './samples/sample_intersection_2.jpg',
  },
  {
    id: 'urban-road',
    title: '遠距號誌與標誌路段',
    subtitle: '高密度號誌與遠距小標誌測試',
    url: './samples/sample_urban_road.jpg',
  },
  {
    id: 'signs-limit',
    title: '速限與禁制標誌路段',
    subtitle: '包含少數類別「速限標誌」',
    url: './samples/sample_signs_limit.jpg',
  },
];

export const MAX_VIDEO_DURATION_SEC = 30;
export const CACHE_MIN_CONF = 0.15;
export const NMS_IOU_THRESHOLD = 0.45;
