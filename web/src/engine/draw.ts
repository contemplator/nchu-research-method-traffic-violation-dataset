import { CLASSES } from '../config/models';
import type { Detection } from '../types';

export function filterDetections(
  detections: Detection[],
  confThreshold: number,
  enabledClasses: Record<number, boolean>,
): Detection[] {
  return detections.filter(
    (d) => d.score >= confThreshold && enabledClasses[d.classId] !== false,
  );
}

/**
 * Draws the source image/video frame and filtered detections onto a target canvas.
 */
export function drawDetectionsOnCanvas(
  canvas: HTMLCanvasElement,
  source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  detections: Detection[],
  confThreshold: number,
  enabledClasses: Record<number, boolean>,
  options?: { drawSource?: boolean },
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  if (w === 0 || h === 0) return;

  ctx.clearRect(0, 0, w, h);
  if (options?.drawSource !== false) {
    ctx.drawImage(source, 0, 0, w, h);
  }

  const visible = filterDetections(detections, confThreshold, enabledClasses);
  const lineWidth = Math.max(2, Math.round(w / 320));
  const fontSize = Math.max(12, Math.round(w / 46));
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Noto Sans TC", sans-serif`;
  ctx.textBaseline = 'top';

  for (const det of visible) {
    const cls = CLASSES[det.classId] ?? {
      zhName: `Class ${det.classId}`,
      name: `cls_${det.classId}`,
      color: '#38bdf8',
    };

    const x = det.x1 * w;
    const y = det.y1 * h;
    const bw = (det.x2 - det.x1) * w;
    const bh = (det.y2 - det.y1) * h;

    // Bounding box
    ctx.strokeStyle = cls.color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, bw, bh);

    // Subtle tinted fill
    ctx.fillStyle = `${cls.color}22`;
    ctx.fillRect(x, y, bw, bh);

    // Label pill
    const label = `${cls.zhName} ${(det.score * 100).toFixed(0)}%`;
    const textMetrics = ctx.measureText(label);
    const padX = 5;
    const padY = 3;
    const pillW = textMetrics.width + padX * 2;
    const pillH = fontSize + padY * 2;
    const pillY = y - pillH >= 0 ? y - pillH : y;

    ctx.fillStyle = cls.color;
    ctx.fillRect(x, pillY, pillW, pillH);

    ctx.fillStyle = '#0f172a';
    ctx.fillText(label, x + padX, pillY + padY);
  }
}
