import Link from "next/link";
import GradCheck from "@/components/GradCheck";

const sections = [
  {
    title: "1 · A neuron is just a weighted sum",
    body: `Each dense layer stores a weight matrix W (inputs × units) and a bias vector b.
Given a batch of inputs A, the layer computes Z = A·W + b and then applies a non-linear
function f to get its output A' = f(Z). Without f, stacking layers would collapse into a
single linear map — which is why a "no hidden layers" network can only draw straight lines.`,
    code: `Z[r][o] = b[o] + Σᵢ A[r][i] · W[i][o]
A'[r][o] = f(Z[r][o])`,
  },
  {
    title: "2 · Softmax turns scores into probabilities",
    body: `The last layer outputs raw scores (logits). Softmax exponentiates and normalises them
so they sum to 1. We subtract the row max first, otherwise exp() overflows.`,
    code: `p[c] = exp(z[c] - max(z)) / Σⱼ exp(z[j] - max(z))`,
  },
  {
    title: "3 · Cross-entropy measures surprise",
    body: `The loss is the negative log-probability the model assigned to the correct class.
If the model says "90% class 2" and the answer is class 2, the loss is -log(0.9) ≈ 0.105.
If it says 1%, the loss is 4.6 — a much bigger penalty.`,
    code: `L = -(1/N) Σ log p[correct class]`,
  },
  {
    title: "4 · Backpropagation is the chain rule, applied backwards",
    body: `The beautiful part: the gradient of softmax + cross-entropy with respect to the logits
simplifies to (p − y). From there we walk backwards through the layers, and each one hands the
gradient to the layer before it.`,
    code: `dZ      = dA ⊙ f'(Z)
dW      = A_prevᵀ · dZ
db      = colsum(dZ)
dA_prev = dZ · Wᵀ`,
  },
  {
    title: "5 · Optimizers decide how far to step",
    body: `SGD moves straight down the gradient. Momentum accumulates velocity so it powers through
flat regions. Adam keeps a running average of both the gradient and its square, giving every
single weight its own adaptive learning rate — which is why it usually converges fastest here.`,
    code: `SGD      : w -= lr · g
Momentum : v = μv - lr·g ;  w += v
Adam     : m = β₁m + (1-β₁)g
           v = β₂v + (1-β₂)g²
           w -= lr · m̂ / (√v̂ + ε)`,
  },
  {
    title: "6 · Initialisation and regularisation",
    body: `Weights start from a Gaussian scaled by He (for ReLU) or Xavier (for tanh/sigmoid)
so signals neither explode nor vanish as they pass through layers. L2 weight decay adds λ·w to
every gradient, shrinking weights toward zero and keeping the decision boundary smooth —
watch the "overfit gap" stat drop when you enable it.`,
    code: `scale_He     = √(2 / fan_in)
scale_Xavier = √(2 / (fan_in + fan_out))
g += λ · w   // L2`,
  },
];

const experiments = [
  {
    q: "Can a network with zero hidden layers solve XOR?",
    a: "No — remove all hidden layers on the XOR dataset and accuracy sticks near 50%. Now add the x₁x₂ feature and it jumps to ~100%. That single feature makes the problem linearly separable.",
  },
  {
    q: "What does a learning rate of 0.3 do?",
    a: "On the spiral dataset the loss curve turns into a jagged mountain range — the optimizer overshoots the minimum every step. Drop to 0.01 and it becomes a smooth slide.",
  },
  {
    q: "How do I make a model overfit on purpose?",
    a: "Set noise to 0.3, samples to 100, use 5 hidden layers of 16 neurons and train for a few thousand epochs. Train accuracy goes to 100% while test accuracy stalls — that's the overfit gap turning amber.",
  },
  {
    q: "Why is ReLU sometimes worse than tanh here?",
    a: "ReLU units can die (output 0 for every input, gradient 0 forever). With only 8 neurons on the spiral, losing two of them hurts. LeakyReLU keeps a 0.01 slope so they can recover.",
  },
];

export default function LearnPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-8 pb-10">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">
          The math behind the pixels
        </p>
        <h1 className="mt-2 text-3xl font-black tracking-tight">
          How NeuralForge actually works
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          Nothing here is a black box. This page walks through every equation implemented in{" "}
          <code className="rounded bg-white/5 px-1.5 py-0.5 text-cyan-300">
            src/lib/nn/engine.ts
          </code>{" "}
          — roughly 300 lines of TypeScript that train a real neural network in your browser.
        </p>
      </header>

      <div className="space-y-4">
        {sections.map((s) => (
          <article
            key={s.title}
            className="rounded-2xl border border-white/10 bg-white/[0.03] p-5"
          >
            <h2 className="text-base font-bold text-slate-100">{s.title}</h2>
            <p className="mt-2 whitespace-pre-line text-sm leading-relaxed text-slate-400">
              {s.body}
            </p>
            <pre className="mt-3 overflow-x-auto rounded-xl border border-white/10 bg-slate-950/70 p-3 text-[12px] leading-relaxed text-cyan-200">
              {s.code}
            </pre>
          </article>
        ))}
      </div>

      <GradCheck />

      <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
        <h2 className="text-base font-bold">Experiments to try in the lab</h2>
        <dl className="mt-3 space-y-4">
          {experiments.map((e) => (
            <div key={e.q}>
              <dt className="text-sm font-semibold text-cyan-300">{e.q}</dt>
              <dd className="mt-1 text-sm leading-relaxed text-slate-400">{e.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <Link
        href="/"
        className="inline-block rounded-xl bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300"
      >
        ← Back to the lab
      </Link>
    </div>
  );
}
