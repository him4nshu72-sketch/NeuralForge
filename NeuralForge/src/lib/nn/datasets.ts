import { gaussian, mulberry32 } from "./random";

export type DatasetName = "spiral" | "circles" | "moons" | "xor" | "blobs";

export const DATASETS: { id: DatasetName; label: string; classes: number; blurb: string }[] = [
  { id: "spiral", label: "Spiral", classes: 3, blurb: "3 interleaved arms — the classic non-linear killer." },
  { id: "circles", label: "Circles", classes: 2, blurb: "A ring inside a ring. Impossible for a linear model." },
  { id: "moons", label: "Moons", classes: 2, blurb: "Two crescents that almost touch." },
  { id: "xor", label: "XOR", classes: 2, blurb: "The problem that killed the perceptron in 1969." },
  { id: "blobs", label: "Blobs", classes: 4, blurb: "Four Gaussian clusters — an easy warm-up." },
];

export interface Point {
  x: number; // in [-1, 1]
  y: number; // in [-1, 1]
  label: number;
}

export interface Dataset {
  points: Point[];
  classes: number;
}

export function classesOf(name: DatasetName): number {
  return DATASETS.find((d) => d.id === name)?.classes ?? 2;
}

export function generateDataset(
  name: DatasetName,
  count: number,
  noise: number,
  seed = 42,
): Dataset {
  const rng = mulberry32(seed);
  const points: Point[] = [];
  const classes = classesOf(name);
  const jitter = () => gaussian(rng, 0, noise);

  switch (name) {
    case "spiral": {
      const perClass = Math.floor(count / 3);
      for (let c = 0; c < 3; c++) {
        for (let i = 0; i < perClass; i++) {
          const r = (i / perClass) * 0.95;
          const t = ((c * 2 * Math.PI) / 3) + (i / perClass) * 3.2 + jitter() * 1.2;
          points.push({
            x: clamp(r * Math.sin(t) + jitter() * 0.35),
            y: clamp(r * Math.cos(t) + jitter() * 0.35),
            label: c,
          });
        }
      }
      break;
    }
    case "circles": {
      const half = Math.floor(count / 2);
      for (let i = 0; i < half; i++) {
        const a = rng() * 2 * Math.PI;
        const r = rng() * 0.4;
        points.push({ x: clamp(r * Math.cos(a) + jitter()), y: clamp(r * Math.sin(a) + jitter()), label: 0 });
      }
      for (let i = 0; i < count - half; i++) {
        const a = rng() * 2 * Math.PI;
        const r = 0.65 + rng() * 0.3;
        points.push({ x: clamp(r * Math.cos(a) + jitter()), y: clamp(r * Math.sin(a) + jitter()), label: 1 });
      }
      break;
    }
    case "moons": {
      const half = Math.floor(count / 2);
      for (let i = 0; i < half; i++) {
        const a = Math.PI * (i / half);
        points.push({
          x: clamp(Math.cos(a) * 0.8 - 0.2 + jitter()),
          y: clamp(Math.sin(a) * 0.6 - 0.25 + jitter()),
          label: 0,
        });
      }
      for (let i = 0; i < count - half; i++) {
        const a = Math.PI * (i / (count - half));
        points.push({
          x: clamp(1 - Math.cos(a) * 0.8 - 0.8 + jitter()),
          y: clamp(0.25 - Math.sin(a) * 0.6 + jitter()),
          label: 1,
        });
      }
      break;
    }
    case "xor": {
      for (let i = 0; i < count; i++) {
        let x = rng() * 2 - 1;
        let y = rng() * 2 - 1;
        // push points away from the axes so the pattern is readable
        x += x > 0 ? 0.06 : -0.06;
        y += y > 0 ? 0.06 : -0.06;
        const label = x * y > 0 ? 1 : 0;
        points.push({ x: clamp(x + jitter()), y: clamp(y + jitter()), label });
      }
      break;
    }
    case "blobs": {
      const centers = [
        [-0.55, -0.55],
        [0.55, -0.55],
        [-0.55, 0.55],
        [0.55, 0.55],
      ];
      for (let i = 0; i < count; i++) {
        const c = i % 4;
        points.push({
          x: clamp(centers[c][0] + gaussian(rng, 0, 0.16 + noise)),
          y: clamp(centers[c][1] + gaussian(rng, 0, 0.16 + noise)),
          label: c,
        });
      }
      break;
    }
  }

  return { points, classes };
}

function clamp(v: number): number {
  return Math.max(-1, Math.min(1, v));
}

/* ------------------------------------------------------------------ */
/* Feature engineering                                                 */
/* ------------------------------------------------------------------ */

export type FeatureId = "x1" | "x2" | "x1sq" | "x2sq" | "x1x2" | "sinx1" | "sinx2" | "r";

export const FEATURES: { id: FeatureId; label: string; fn: (x: number, y: number) => number }[] = [
  { id: "x1", label: "x₁", fn: (x) => x },
  { id: "x2", label: "x₂", fn: (_x, y) => y },
  { id: "x1sq", label: "x₁²", fn: (x) => x * x },
  { id: "x2sq", label: "x₂²", fn: (_x, y) => y * y },
  { id: "x1x2", label: "x₁x₂", fn: (x, y) => x * y },
  { id: "sinx1", label: "sin(3x₁)", fn: (x) => Math.sin(3 * x) },
  { id: "sinx2", label: "sin(3x₂)", fn: (_x, y) => Math.sin(3 * y) },
  { id: "r", label: "√(x₁²+x₂²)", fn: (x, y) => Math.sqrt(x * x + y * y) },
];

export function featureLabel(id: string): string {
  return FEATURES.find((f) => f.id === id)?.label ?? id;
}

/** Turn raw 2-D points into the flat design matrix the network consumes. */
export function buildMatrix(
  points: Point[],
  features: FeatureId[],
): { X: Float64Array; y: Int32Array; rows: number; cols: number } {
  const fns = features.map((id) => FEATURES.find((f) => f.id === id)!.fn);
  const cols = fns.length;
  const rows = points.length;
  const X = new Float64Array(rows * cols);
  const y = new Int32Array(rows);
  for (let r = 0; r < rows; r++) {
    const p = points[r];
    for (let c = 0; c < cols; c++) X[r * cols + c] = fns[c](p.x, p.y);
    y[r] = p.label;
  }
  return { X, y, rows, cols };
}

export function encodeGrid(
  resolution: number,
  features: FeatureId[],
): { X: Float64Array; rows: number; cols: number } {
  const fns = features.map((id) => FEATURES.find((f) => f.id === id)!.fn);
  const cols = fns.length;
  const rows = resolution * resolution;
  const X = new Float64Array(rows * cols);
  let r = 0;
  for (let j = 0; j < resolution; j++) {
    const y = 1 - (2 * j) / (resolution - 1);
    for (let i = 0; i < resolution; i++) {
      const x = -1 + (2 * i) / (resolution - 1);
      for (let c = 0; c < cols; c++) X[r * cols + c] = fns[c](x, y);
      r++;
    }
  }
  return { X, rows, cols };
}

export function splitDataset(
  points: Point[],
  trainRatio: number,
  seed = 7,
): { train: Point[]; test: Point[] } {
  const rng = mulberry32(seed);
  const shuffled = points.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const cut = Math.max(1, Math.floor(shuffled.length * trainRatio));
  return { train: shuffled.slice(0, cut), test: shuffled.slice(cut) };
}
