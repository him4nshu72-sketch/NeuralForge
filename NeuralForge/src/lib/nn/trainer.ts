import {
  buildMatrix,
  classesOf,
  encodeGrid,
  generateDataset,
  splitDataset,
  type DatasetName,
  type FeatureId,
  type Point,
} from "./datasets";
import { MLP, type Activation, type OptimizerName } from "./engine";
import { mulberry32 } from "./random";

export interface TrainConfig {
  dataset: DatasetName;
  samples: number;
  noise: number;
  trainRatio: number;
  features: FeatureId[];
  hidden: number[];
  activation: Activation;
  optimizer: OptimizerName;
  learningRate: number;
  batchSize: number;
  l2: number;
  seed: number;
}

export const DEFAULT_CONFIG: TrainConfig = {
  dataset: "spiral",
  samples: 450,
  noise: 0.06,
  trainRatio: 0.7,
  features: ["x1", "x2"],
  hidden: [8, 8],
  activation: "tanh",
  optimizer: "adam",
  learningRate: 0.03,
  batchSize: 32,
  l2: 0,
  seed: 1337,
};

export interface Metrics {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAcc: number;
  testAcc: number;
}

export interface BoundaryGrid {
  size: number;
  classes: Uint8Array;
  conf: Float32Array;
}

export class Trainer {
  readonly config: TrainConfig;
  readonly classes: number;
  readonly model: MLP;
  readonly trainPoints: Point[];
  readonly testPoints: Point[];

  private trainX: Float64Array;
  private trainY: Int32Array;
  private testX: Float64Array;
  private testY: Int32Array;
  private cols: number;
  private order: Int32Array;
  private rng: () => number;

  epoch = 0;
  history: Metrics[] = [];
  latest: Metrics;

  private gridCache = new Map<number, { X: Float64Array; rows: number }>();

  constructor(config: TrainConfig) {
    this.config = config;
    this.classes = classesOf(config.dataset);
    const { points } = generateDataset(config.dataset, config.samples, config.noise, config.seed);
    const { train, test } = splitDataset(points, config.trainRatio, config.seed + 1);
    this.trainPoints = train;
    this.testPoints = test;

    const tr = buildMatrix(train, config.features);
    const te = buildMatrix(test, config.features);
    this.trainX = tr.X;
    this.trainY = tr.y;
    this.testX = te.X;
    this.testY = te.y;
    this.cols = tr.cols;

    this.model = new MLP({
      inputSize: this.cols,
      hidden: config.hidden,
      outputSize: this.classes,
      activation: config.activation,
      seed: config.seed,
    });

    this.order = new Int32Array(train.length);
    for (let i = 0; i < train.length; i++) this.order[i] = i;
    this.rng = mulberry32(config.seed + 99);

    this.latest = this.computeMetrics();
    this.history.push(this.latest);
  }

  get paramCount(): number {
    return this.model.paramCount;
  }

  /** One full pass over the shuffled training set in mini-batches. */
  runEpoch(): void {
    const n = this.trainY.length;
    if (n === 0) return;
    // Fisher–Yates shuffle of the sample order
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      const t = this.order[i];
      this.order[i] = this.order[j];
      this.order[j] = t;
    }

    const bs = Math.max(1, Math.min(this.config.batchSize, n));
    const cols = this.cols;
    const bx = new Float64Array(bs * cols);
    const by = new Int32Array(bs);

    for (let start = 0; start + bs <= n; start += bs) {
      for (let k = 0; k < bs; k++) {
        const idx = this.order[start + k];
        for (let c = 0; c < cols; c++) bx[k * cols + c] = this.trainX[idx * cols + c];
        by[k] = this.trainY[idx];
      }
      this.model.trainBatch(bx, by, bs, {
        name: this.config.optimizer,
        learningRate: this.config.learningRate,
        l2: this.config.l2,
      });
    }
    this.epoch += 1;
  }

  computeMetrics(): Metrics {
    const train = this.model.evaluate(this.trainX, this.trainY, this.trainY.length);
    const test = this.model.evaluate(this.testX, this.testY, this.testY.length);
    return {
      epoch: this.epoch,
      trainLoss: train.loss,
      testLoss: test.loss,
      trainAcc: train.accuracy,
      testAcc: test.accuracy,
    };
  }

  record(): Metrics {
    const m = this.computeMetrics();
    this.latest = m;
    this.history.push(m);
    if (this.history.length > 4000) this.history.splice(0, 1000);
    return m;
  }

  boundary(size: number): BoundaryGrid {
    let cache = this.gridCache.get(size);
    if (!cache) {
      const g = encodeGrid(size, this.config.features);
      cache = { X: g.X, rows: g.rows };
      this.gridCache.set(size, cache);
    }
    const probs = this.model.predict(cache.X, cache.rows);
    const k = this.classes;
    const classes = new Uint8Array(cache.rows);
    const conf = new Float32Array(cache.rows);
    for (let r = 0; r < cache.rows; r++) {
      const off = r * k;
      let best = 0;
      for (let c = 1; c < k; c++) if (probs[off + c] > probs[off + best]) best = c;
      classes[r] = best;
      conf[r] = probs[off + best];
    }
    return { size, classes, conf };
  }

  /** Sample the history down to ~120 points for storage / charts. */
  compactHistory(limit = 120): Metrics[] {
    if (this.history.length <= limit) return this.history.slice();
    const step = this.history.length / limit;
    const out: Metrics[] = [];
    for (let i = 0; i < limit; i++) out.push(this.history[Math.floor(i * step)]);
    out.push(this.history[this.history.length - 1]);
    return out;
  }
}
