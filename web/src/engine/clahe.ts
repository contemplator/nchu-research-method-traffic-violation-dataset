/**
 * Pure TypeScript LAB-CLAHE implementation matching OpenCV:
 *   lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
 *   l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
 *   out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
 *
 * Operates in-place on RGBA ImageData pixels (preserving alpha).
 */

const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

const LAB_DELTA = 6 / 29;
const LAB_DELTA_SQ = LAB_DELTA * LAB_DELTA;
const LAB_DELTA_CUBE = LAB_DELTA * LAB_DELTA * LAB_DELTA;

function labF(t: number): number {
  return t > LAB_DELTA_CUBE ? Math.cbrt(t) : t / (3 * LAB_DELTA_SQ) + 4 / 29;
}

function labFInv(t: number): number {
  return t > LAB_DELTA ? t * t * t : 3 * LAB_DELTA_SQ * (t - 4 / 29);
}

function linearToSrgbByte(c: number): number {
  if (c <= 0) return 0;
  if (c >= 1) return 255;
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  const b = Math.round(v * 255);
  return b < 0 ? 0 : b > 255 ? 255 : b;
}

/**
 * Apply 2D CLAHE on an 8-bit single-channel plane (OpenCV-compatible tile grid & bilinear interpolation).
 */
function applyClaheChannel(
  src: Uint8Array,
  width: number,
  height: number,
  clipLimit = 2.0,
  tileGridX = 8,
  tileGridY = 8,
): Uint8Array {
  const tileW = width / tileGridX;
  const tileH = height / tileGridY;
  const numTiles = tileGridX * tileGridY;
  const luts = new Uint8Array(numTiles * 256);
  const hist = new Int32Array(256);

  for (let ty = 0; ty < tileGridY; ty++) {
    const y0 = Math.floor(ty * tileH);
    const y1 = ty === tileGridY - 1 ? height : Math.floor((ty + 1) * tileH);
    for (let tx = 0; tx < tileGridX; tx++) {
      const x0 = Math.floor(tx * tileW);
      const x1 = tx === tileGridX - 1 ? width : Math.floor((tx + 1) * tileW);

      hist.fill(0);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        const row = y * width;
        for (let x = x0; x < x1; x++) {
          hist[src[row + x]]++;
          count++;
        }
      }

      if (count === 0) continue;

      const limit = Math.max(1, Math.floor((clipLimit * count) / 256));
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }

      const avgInc = Math.floor(excess / 256);
      const rem = excess - avgInc * 256;
      for (let i = 0; i < 256; i++) {
        hist[i] += avgInc + (i < rem ? 1 : 0);
      }

      const lutOffset = (ty * tileGridX + tx) * 256;
      let cdf = 0;
      const scale = 255 / count;
      for (let i = 0; i < 256; i++) {
        cdf += hist[i];
        const v = Math.round(cdf * scale);
        luts[lutOffset + i] = v > 255 ? 255 : v;
      }
    }
  }

  const dst = new Uint8Array(width * height);
  const invTileW = 1 / tileW;
  const invTileH = 1 / tileH;

  for (let y = 0; y < height; y++) {
    const fy = (y + 0.5) * invTileH - 0.5;
    let ty0 = Math.floor(fy);
    let ty1 = ty0 + 1;
    let wy = fy - ty0;
    if (ty0 < 0) {
      ty0 = 0;
      ty1 = 0;
      wy = 0;
    } else if (ty1 >= tileGridY) {
      ty0 = tileGridY - 1;
      ty1 = tileGridY - 1;
      wy = 0;
    }
    const wy0 = 1 - wy;
    const row = y * width;

    for (let x = 0; x < width; x++) {
      const fx = (x + 0.5) * invTileW - 0.5;
      let tx0 = Math.floor(fx);
      let tx1 = tx0 + 1;
      let wx = fx - tx0;
      if (tx0 < 0) {
        tx0 = 0;
        tx1 = 0;
        wx = 0;
      } else if (tx1 >= tileGridX) {
        tx0 = tileGridX - 1;
        tx1 = tileGridX - 1;
        wx = 0;
      }
      const wx0 = 1 - wx;
      const val = src[row + x];

      const v00 = luts[(ty0 * tileGridX + tx0) * 256 + val];
      const v10 = luts[(ty0 * tileGridX + tx1) * 256 + val];
      const v01 = luts[(ty1 * tileGridX + tx0) * 256 + val];
      const v11 = luts[(ty1 * tileGridX + tx1) * 256 + val];

      const top = v00 * wx0 + v10 * wx;
      const bot = v01 * wx0 + v11 * wx;
      dst[row + x] = Math.round(top * wy0 + bot * wy);
    }
  }

  return dst;
}

/**
 * Applies LAB-CLAHE (clipLimit=2.0, tileGridSize=8x8) in-place on RGBA ImageData.
 */
export function applyLabClaheInPlace(
  imageData: ImageData,
  clipLimit = 2.0,
  tileSize = 8,
): void {
  const { data, width, height } = imageData;
  const n = width * height;
  const lPlane = new Uint8Array(n);
  const aPlane = new Float32Array(n);
  const bPlane = new Float32Array(n);

  // D65 reference white
  const Xn = 0.950456;
  const Yn = 1.0;
  const Zn = 1.088754;

  // RGB -> CIE LAB (OpenCV L scaled 0..255)
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = SRGB_TO_LINEAR[data[p]];
    const g = SRGB_TO_LINEAR[data[p + 1]];
    const b = SRGB_TO_LINEAR[data[p + 2]];

    const X = (0.412453 * r + 0.357580 * g + 0.180423 * b) / Xn;
    const Y = (0.212671 * r + 0.715160 * g + 0.072169 * b) / Yn;
    const Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / Zn;

    const fx = labF(X);
    const fy = labF(Y);
    const fz = labF(Z);

    const L = 116 * fy - 16; // 0..100
    const lByte = Math.round(L * 2.55);
    lPlane[i] = lByte < 0 ? 0 : lByte > 255 ? 255 : lByte;
    aPlane[i] = 500 * (fx - fy);
    bPlane[i] = 200 * (fy - fz);
  }

  // CLAHE on L channel
  const lEq = applyClaheChannel(lPlane, width, height, clipLimit, tileSize, tileSize);

  // CIE LAB -> RGB
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const L = lEq[i] / 2.55;
    const fy = (L + 16) / 116;
    const fx = fy + aPlane[i] / 500;
    const fz = fy - bPlane[i] / 200;

    const X = Xn * labFInv(fx);
    const Y = Yn * labFInv(fy);
    const Z = Zn * labFInv(fz);

    const rLin = 3.240479 * X - 1.537150 * Y - 0.498535 * Z;
    const gLin = -0.969256 * X + 1.875992 * Y + 0.041556 * Z;
    const bLin = 0.055648 * X - 0.204043 * Y + 1.057311 * Z;

    data[p] = linearToSrgbByte(rLin);
    data[p + 1] = linearToSrgbByte(gLin);
    data[p + 2] = linearToSrgbByte(bLin);
  }
}
