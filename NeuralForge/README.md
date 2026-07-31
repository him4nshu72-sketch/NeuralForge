# 🧠 NeuralForge

**An interactive deep-learning lab. The neural network is written from scratch in TypeScript — no TensorFlow, no PyTorch, no math libraries.**

Design an architecture, press *Train*, and watch the decision boundary redraw itself 60 times a second while the loss curve falls. Save your best runs to a PostgreSQL-backed leaderboard.

![status](https://img.shields.io/badge/ML%20libraries%20used-0-brightgreen) ![stack](https://img.shields.io/badge/stack-Next.js%20·%20TypeScript%20·%20Postgres-blue)

---

## Why this project is interesting

Most "AI projects" are 20 lines of `model.fit()`. This one implements the actual math:

| Implemented by hand | Where |
| --- | --- |
| Dense layers with flat `Float64Array` matrices | `src/lib/nn/engine.ts` |
| Forward pass `Z = A·W + b`, activations (ReLU, LeakyReLU, tanh, sigmoid) | `engine.ts` |
| Numerically-stable softmax + categorical cross-entropy | `engine.ts` |
| Reverse-mode backpropagation (`dZ = dA ⊙ f'(Z)`, `dW = Aᵀ·dZ`, …) | `engine.ts` |
| Optimizers: **SGD**, **Momentum**, **Adam** (with bias correction) | `engine.ts` |
| He / Xavier weight initialisation, L2 weight decay | `engine.ts` |
| Mini-batch shuffling & epoch loop | `src/lib/nn/trainer.ts` |
| Synthetic datasets: spiral, circles, moons, XOR, blobs | `src/lib/nn/datasets.ts` |
| Feature engineering (`x₁²`, `x₁x₂`, `sin 3x₁`, radius…) | `datasets.ts` |
| **Finite-difference gradient checking** (proof of correctness) | `src/lib/nn/gradcheck.ts` |

Everything trains **in the browser** on the main thread at interactive frame rates.

## Features

- 🎛 **Full control panel** — dataset, noise, train/test split, hidden layers & neurons, activation, optimizer, learning rate, batch size, L2.
- 🌈 **Live decision-boundary heatmap** — every pixel is a real forward pass; opacity encodes the model's confidence.
- 📉 **Live loss & accuracy charts** for both train and test sets, plus an "overfit gap" indicator.
- 🕸 **Live network diagram** — each edge is a real weight; blue = positive, orange = negative, thickness = magnitude.
- 🏆 **Leaderboard** — save a run to Postgres with its full hyper-parameter config and a thumbnail of the learned decision surface, ranked by test accuracy.
- 🧪 **Gradient checker** on the *How it works* page — compares analytic gradients to numerical ones and reports the max relative error (≈1e-9).
- 📖 **Explainer page** deriving every equation in the codebase.

## Tech stack

- **Next.js (App Router)** + React 19 + TypeScript
- **Tailwind CSS v4**
- **PostgreSQL** with **Drizzle ORM**
- Canvas 2D + SVG for all visualisations (no chart library)

## Getting started

```bash
npm install
echo "DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/app_db" > .env
npx drizzle-kit push     # create the experiments table
npm run dev
```

Open http://localhost:3000

## API

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/api/experiments?sort=acc\|recent&limit=50` | List saved runs |
| `POST` | `/api/experiments` | Save a run (config + metrics + boundary thumbnail) |
| `GET` | `/api/experiments/:id` | Fetch one run |
| `DELETE` | `/api/experiments/:id` | Delete a run |
| `GET` | `/api/health` | Health check |

## Project structure

```
src/
├── app/
│   ├── page.tsx              # the lab
│   ├── leaderboard/          # ranked experiments (server component + Drizzle)
│   ├── learn/                # the math, derived step by step
│   └── api/experiments/      # REST endpoints
├── components/               # canvas + SVG visualisations, control panels
├── db/                       # Drizzle schema & client
└── lib/nn/
    ├── engine.ts             # ⭐ the neural network
    ├── trainer.ts            # data pipeline + epoch loop
    ├── datasets.ts           # synthetic data + feature engineering
    ├── gradcheck.ts          # correctness proof
    ├── palette.ts            # boundary rendering
    └── random.ts             # seeded PRNG (reproducible runs)
```

## Things to try

1. Remove every hidden layer on **XOR** → stuck at ~50%. Add the `x₁x₂` feature → ~100%. That is the 1969 perceptron problem, solved with feature engineering.
2. Set the learning rate to `0.3` on **Spiral** and watch the loss curve become a mountain range.
3. Noise `0.3` + 100 samples + 5×16 layers → train accuracy 100%, test accuracy 70%. Textbook overfitting. Now turn on L2.

---

Built as a from-scratch exploration of how neural networks actually learn.
