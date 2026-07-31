import { MLP, type Activation } from "./engine";
import { mulberry32 } from "./random";

export interface GradCheckResult {
  activation: Activation;
  samples: number;
  maxRelativeError: number;
  meanRelativeError: number;
  passed: boolean;
}

/**
 * Finite-difference gradient checking — the standard way to prove a
 * hand-written backprop implementation is correct.
 *
 *   numeric  = (L(w + ε) − L(w − ε)) / 2ε
 *   relative = |analytic − numeric| / max(1e-8, |analytic| + |numeric|)
 *
 * Anything below ~1e-5 means the analytic gradients match calculus.
 */
export function gradientCheck(activation: Activation, seed = 3): GradCheckResult {
  const rng = mulberry32(seed);
  const inputSize = 4;
  const classes = 3;
  const batch = 12;

  const X = new Float64Array(batch * inputSize);
  for (let i = 0; i < X.length; i++) X[i] = rng() * 2 - 1;
  const y = new Int32Array(batch);
  for (let i = 0; i < batch; i++) y[i] = Math.floor(rng() * classes);

  const model = new MLP({
    inputSize,
    hidden: [5, 4],
    outputSize: classes,
    activation,
    seed: seed + 11,
  });

  model.computeGradients(X, y, batch);

  const eps = 1e-5;
  let maxRel = 0;
  let sumRel = 0;
  let checked = 0;

  for (let li = 0; li < model.layers.length; li++) {
    const { W, dW } = model.parameterView(li);
    // sample a handful of weights per layer instead of all of them
    for (let s = 0; s < 12; s++) {
      const idx = Math.floor(rng() * W.length);
      const original = W[idx];

      W[idx] = original + eps;
      const lossPlus = model.lossOn(X, y, batch);
      W[idx] = original - eps;
      const lossMinus = model.lossOn(X, y, batch);
      W[idx] = original;

      const numeric = (lossPlus - lossMinus) / (2 * eps);
      const analytic = dW[idx];
      const rel =
        Math.abs(analytic - numeric) /
        Math.max(1e-8, Math.abs(analytic) + Math.abs(numeric));

      maxRel = Math.max(maxRel, rel);
      sumRel += rel;
      checked++;
    }
  }

  return {
    activation,
    samples: checked,
    maxRelativeError: maxRel,
    meanRelativeError: sumRel / Math.max(1, checked),
    passed: maxRel < 1e-4,
  };
}

export function runAllGradientChecks(): GradCheckResult[] {
  return (["relu", "leakyRelu", "tanh", "sigmoid"] as Activation[]).map((a) =>
    gradientCheck(a),
  );
}
