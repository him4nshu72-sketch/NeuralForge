/**
 * NeuralForge — a tiny neural-network engine written from scratch.
 *
 * No TensorFlow, no PyTorch, no math libraries. Just flat Float64Arrays,
 * hand-derived gradients and three optimizers. Everything here runs in
 * the browser at 60fps.
 *
 * Math recap for a single dense layer:
 *   Z = A_prev · W + b
 *   A = f(Z)
 * Backward pass (reverse-mode differentiation):
 *   dZ      = dA ⊙ f'(Z)
 *   dW      = A_prevᵀ · dZ
 *   db      = colsum(dZ)
 *   dA_prev = dZ · Wᵀ
 * The output layer uses softmax + categorical cross-entropy, whose
 * combined gradient collapses to the famous `dZ = (p - y)`.
 */

import { mulberry32 } from "./random";

export type Activation = "relu" | "leakyRelu" | "tanh" | "sigmoid";
export type OptimizerName = "sgd" | "momentum" | "adam";

export const ACTIVATIONS: Activation[] = ["relu", "leakyRelu", "tanh", "sigmoid"];
export const OPTIMIZERS: OptimizerName[] = ["sgd", "momentum", "adam"];

export interface OptimizerOptions {
  name: OptimizerName;
  learningRate: number;
  momentum?: number; // for "momentum"
  beta1?: number; // for "adam"
  beta2?: number;
  epsilon?: number;
  l2?: number; // weight decay strength
}

function applyActivation(name: Activation, z: number): number {
  switch (name) {
    case "relu":
      return z > 0 ? z : 0;
    case "leakyRelu":
      return z > 0 ? z : 0.01 * z;
    case "tanh":
      return Math.tanh(z);
    case "sigmoid":
      return 1 / (1 + Math.exp(-z));
  }
}

/** Derivative expressed in terms of the *pre*-activation value z. */
function activationPrime(name: Activation, z: number, a: number): number {
  switch (name) {
    case "relu":
      return z > 0 ? 1 : 0;
    case "leakyRelu":
      return z > 0 ? 1 : 0.01;
    case "tanh":
      return 1 - a * a;
    case "sigmoid":
      return a * (1 - a);
  }
}

class DenseLayer {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly activation: Activation | null; // null => linear (logits)

  W: Float64Array;
  b: Float64Array;

  // gradients
  dW: Float64Array;
  db: Float64Array;

  // optimizer state
  mW: Float64Array;
  vW: Float64Array;
  mb: Float64Array;
  vb: Float64Array;

  // forward caches
  private cachedInput: Float64Array | null = null;
  private cachedBatch = 0;
  Z: Float64Array = new Float64Array(0);
  A: Float64Array = new Float64Array(0);

  constructor(
    inputSize: number,
    outputSize: number,
    activation: Activation | null,
    rng: () => number,
  ) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = activation;

    const size = inputSize * outputSize;
    this.W = new Float64Array(size);
    this.b = new Float64Array(outputSize);
    this.dW = new Float64Array(size);
    this.db = new Float64Array(outputSize);
    this.mW = new Float64Array(size);
    this.vW = new Float64Array(size);
    this.mb = new Float64Array(outputSize);
    this.vb = new Float64Array(outputSize);

    // He initialisation for relu-family, Xavier/Glorot otherwise.
    const reluLike = activation === "relu" || activation === "leakyRelu";
    const scale = reluLike
      ? Math.sqrt(2 / inputSize)
      : Math.sqrt(2 / (inputSize + outputSize));
    for (let i = 0; i < size; i++) {
      // Box–Muller inline (avoids importing to keep the hot path tight).
      const u = Math.max(rng(), 1e-12);
      const v = rng();
      this.W[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
    }
  }

  get paramCount(): number {
    return this.W.length + this.b.length;
  }

  forward(input: Float64Array, batch: number): Float64Array {
    const { inputSize: n, outputSize: m } = this;
    if (this.Z.length !== batch * m) {
      this.Z = new Float64Array(batch * m);
      this.A = new Float64Array(batch * m);
    }
    this.cachedInput = input;
    this.cachedBatch = batch;

    const Z = this.Z;
    const A = this.A;
    const W = this.W;

    for (let r = 0; r < batch; r++) {
      const rowIn = r * n;
      const rowOut = r * m;
      for (let o = 0; o < m; o++) {
        let sum = this.b[o];
        for (let i = 0; i < n; i++) {
          sum += input[rowIn + i] * W[i * m + o];
        }
        Z[rowOut + o] = sum;
        A[rowOut + o] = this.activation ? applyActivation(this.activation, sum) : sum;
      }
    }
    return A;
  }

  /**
   * @param dA gradient of loss wrt this layer's activations (batch × outputSize)
   * @returns gradient wrt this layer's inputs (batch × inputSize)
   */
  backward(dA: Float64Array): Float64Array {
    const { inputSize: n, outputSize: m } = this;
    const batch = this.cachedBatch;
    const input = this.cachedInput;
    if (!input) throw new Error("backward() called before forward()");

    const dZ = new Float64Array(batch * m);
    for (let idx = 0; idx < batch * m; idx++) {
      dZ[idx] = this.activation
        ? dA[idx] * activationPrime(this.activation, this.Z[idx], this.A[idx])
        : dA[idx];
    }

    this.dW.fill(0);
    this.db.fill(0);
    const dInput = new Float64Array(batch * n);

    for (let r = 0; r < batch; r++) {
      const rowIn = r * n;
      const rowOut = r * m;
      for (let o = 0; o < m; o++) {
        const g = dZ[rowOut + o];
        if (g === 0) continue;
        this.db[o] += g;
        for (let i = 0; i < n; i++) {
          this.dW[i * m + o] += input[rowIn + i] * g;
          dInput[rowIn + i] += g * this.W[i * m + o];
        }
      }
    }
    return dInput;
  }

  step(opts: OptimizerOptions, t: number): void {
    const lr = opts.learningRate;
    const l2 = opts.l2 ?? 0;
    const beta1 = opts.beta1 ?? 0.9;
    const beta2 = opts.beta2 ?? 0.999;
    const eps = opts.epsilon ?? 1e-8;
    const mu = opts.momentum ?? 0.9;

    const update = (
      params: Float64Array,
      grads: Float64Array,
      m: Float64Array,
      v: Float64Array,
      decay: boolean,
    ) => {
      for (let i = 0; i < params.length; i++) {
        let g = grads[i];
        if (decay && l2 > 0) g += l2 * params[i];

        switch (opts.name) {
          case "sgd":
            params[i] -= lr * g;
            break;
          case "momentum":
            m[i] = mu * m[i] - lr * g;
            params[i] += m[i];
            break;
          case "adam": {
            m[i] = beta1 * m[i] + (1 - beta1) * g;
            v[i] = beta2 * v[i] + (1 - beta2) * g * g;
            const mHat = m[i] / (1 - Math.pow(beta1, t));
            const vHat = v[i] / (1 - Math.pow(beta2, t));
            params[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
            break;
          }
        }
      }
    };

    update(this.W, this.dW, this.mW, this.vW, true);
    update(this.b, this.db, this.mb, this.vb, false);
  }
}

export interface MLPConfig {
  inputSize: number;
  hidden: number[];
  outputSize: number;
  activation: Activation;
  seed?: number;
}

export class MLP {
  readonly layers: DenseLayer[] = [];
  readonly config: MLPConfig;
  private t = 0;

  constructor(config: MLPConfig) {
    this.config = config;
    const rng = mulberry32(config.seed ?? 1337);
    let prev = config.inputSize;
    for (const units of config.hidden) {
      this.layers.push(new DenseLayer(prev, units, config.activation, rng));
      prev = units;
    }
    // final layer produces raw logits, softmax is applied by the loss
    this.layers.push(new DenseLayer(prev, config.outputSize, null, rng));
  }

  get paramCount(): number {
    return this.layers.reduce((sum, l) => sum + l.paramCount, 0);
  }

  /** Returns logits (batch × outputSize). */
  forwardLogits(X: Float64Array, batch: number): Float64Array {
    let out = X;
    for (const layer of this.layers) out = layer.forward(out, batch);
    return out;
  }

  /** Numerically stable row-wise softmax (in place on a copy). */
  static softmax(logits: Float64Array, batch: number, classes: number): Float64Array {
    const probs = new Float64Array(logits.length);
    for (let r = 0; r < batch; r++) {
      const off = r * classes;
      let max = -Infinity;
      for (let c = 0; c < classes; c++) max = Math.max(max, logits[off + c]);
      let sum = 0;
      for (let c = 0; c < classes; c++) {
        const e = Math.exp(logits[off + c] - max);
        probs[off + c] = e;
        sum += e;
      }
      for (let c = 0; c < classes; c++) probs[off + c] /= sum;
    }
    return probs;
  }

  predict(X: Float64Array, batch: number): Float64Array {
    const logits = this.forwardLogits(X, batch);
    return MLP.softmax(logits, batch, this.config.outputSize);
  }

  /**
   * Forward + backward pass. Fills every layer's dW/db but does NOT update
   * the weights, so it can be used by the gradient checker too.
   * Returns the mean cross-entropy loss of the batch.
   */
  computeGradients(X: Float64Array, y: Int32Array, batch: number): number {
    const classes = this.config.outputSize;
    const logits = this.forwardLogits(X, batch);
    const probs = MLP.softmax(logits, batch, classes);

    let loss = 0;
    const dLogits = new Float64Array(probs.length);
    for (let r = 0; r < batch; r++) {
      const off = r * classes;
      const target = y[r];
      loss -= Math.log(Math.max(probs[off + target], 1e-12));
      // d(softmax + cross-entropy)/d(logits) = (p - onehot)
      for (let c = 0; c < classes; c++) {
        dLogits[off + c] = (probs[off + c] - (c === target ? 1 : 0)) / batch;
      }
    }

    let grad: Float64Array = dLogits;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }
    return loss / batch;
  }

  /** Mean cross-entropy without touching gradients (used by finite differences). */
  lossOn(X: Float64Array, y: Int32Array, batch: number): number {
    const classes = this.config.outputSize;
    const probs = this.predict(X, batch);
    let loss = 0;
    for (let r = 0; r < batch; r++) {
      loss -= Math.log(Math.max(probs[r * classes + y[r]], 1e-12));
    }
    return loss / batch;
  }

  /** One gradient-descent step on a mini-batch. Returns the batch loss. */
  trainBatch(
    X: Float64Array,
    y: Int32Array,
    batch: number,
    opts: OptimizerOptions,
  ): number {
    const loss = this.computeGradients(X, y, batch);
    this.t += 1;
    for (const layer of this.layers) layer.step(opts, this.t);
    return loss;
  }

  /** Direct access to a layer's parameter/gradient buffers (gradient checking). */
  parameterView(layerIndex: number): { W: Float64Array; dW: Float64Array; b: Float64Array; db: Float64Array } {
    const l = this.layers[layerIndex];
    return { W: l.W, dW: l.dW, b: l.b, db: l.db };
  }

  evaluate(
    X: Float64Array,
    y: Int32Array,
    batch: number,
  ): { loss: number; accuracy: number } {
    if (batch === 0) return { loss: 0, accuracy: 0 };
    const classes = this.config.outputSize;
    const probs = this.predict(X, batch);
    let loss = 0;
    let correct = 0;
    for (let r = 0; r < batch; r++) {
      const off = r * classes;
      loss -= Math.log(Math.max(probs[off + y[r]], 1e-12));
      let best = 0;
      for (let c = 1; c < classes; c++) {
        if (probs[off + c] > probs[off + best]) best = c;
      }
      if (best === y[r]) correct += 1;
    }
    return { loss: loss / batch, accuracy: correct / batch };
  }

  /** Flattened first-layer weights, used by the network visualiser. */
  weightsOf(layerIndex: number): { rows: number; cols: number; data: Float64Array } {
    const layer = this.layers[layerIndex];
    return { rows: layer.inputSize, cols: layer.outputSize, data: layer.W };
  }
}
