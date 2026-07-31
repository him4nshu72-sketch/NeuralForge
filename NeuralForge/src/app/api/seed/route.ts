import { NextResponse } from "next/server";
import { sql } from "drizzle-orm";
import { db } from "@/db";
import { experiments, type NewExperiment } from "@/db/schema";
import { DEFAULT_CONFIG, Trainer, type TrainConfig } from "@/lib/nn/trainer";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const SEEDS: { name: string; author: string; epochs: number; config: Partial<TrainConfig> }[] = [
  {
    name: "Adam cracks the spiral",
    author: "neuralforge",
    epochs: 260,
    config: { dataset: "spiral", hidden: [12, 10], activation: "tanh", optimizer: "adam", learningRate: 0.03 },
  },
  {
    name: "XOR solved with one crafted feature",
    author: "neuralforge",
    epochs: 90,
    config: {
      dataset: "xor",
      hidden: [],
      features: ["x1", "x2", "x1x2"],
      optimizer: "adam",
      learningRate: 0.1,
    },
  },
  {
    name: "Circles + radius feature (linear!)",
    author: "neuralforge",
    epochs: 90,
    config: { dataset: "circles", hidden: [], features: ["x1", "x2", "r"], optimizer: "adam", learningRate: 0.1 },
  },
  {
    name: "Deep ReLU moons",
    author: "neuralforge",
    epochs: 160,
    config: { dataset: "moons", hidden: [8, 8, 8], activation: "relu", optimizer: "momentum", learningRate: 0.03 },
  },
  {
    name: "Plain SGD blobs baseline",
    author: "neuralforge",
    epochs: 120,
    config: { dataset: "blobs", hidden: [4], activation: "relu", optimizer: "sgd", learningRate: 0.1 },
  },
];

export async function POST() {
  const [{ count }] = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(experiments);

  if (count > 0) {
    return NextResponse.json({ seeded: 0, message: "Leaderboard already has runs." });
  }

  const rows: NewExperiment[] = [];

  for (const seed of SEEDS) {
    const config: TrainConfig = { ...DEFAULT_CONFIG, ...seed.config };
    const trainer = new Trainer(config);
    for (let e = 0; e < seed.epochs; e++) {
      trainer.runEpoch();
      if (e % 4 === 0) trainer.record();
    }
    const m = trainer.record();
    const thumb = trainer.boundary(26);

    rows.push({
      name: seed.name,
      author: seed.author,
      dataset: config.dataset,
      noise: config.noise,
      samples: config.samples,
      hidden: config.hidden,
      features: config.features,
      activation: config.activation,
      optimizer: config.optimizer,
      learningRate: config.learningRate,
      batchSize: config.batchSize,
      l2: config.l2,
      epochs: m.epoch,
      paramCount: trainer.paramCount,
      trainLoss: m.trainLoss,
      testLoss: m.testLoss,
      trainAcc: m.trainAcc,
      testAcc: m.testAcc,
      history: trainer.compactHistory(80),
      thumb: {
        size: thumb.size,
        classes: Array.from(thumb.classes),
        conf: Array.from(thumb.conf).map((c) => Math.round(c * 100) / 100),
      },
    });
  }

  await db.insert(experiments).values(rows);
  return NextResponse.json({ seeded: rows.length });
}
