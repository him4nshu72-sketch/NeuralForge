import { NextResponse } from "next/server";
import { desc } from "drizzle-orm";
import { db } from "@/db";
import { experiments, type NewExperiment } from "@/db/schema";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Math.min(Number(searchParams.get("limit") ?? 50) || 50, 200);
  const sort = searchParams.get("sort") ?? "acc";

  const rows = await db
    .select()
    .from(experiments)
    .orderBy(sort === "recent" ? desc(experiments.createdAt) : desc(experiments.testAcc))
    .limit(limit);

  return NextResponse.json({ experiments: rows });
}

function num(v: unknown, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const thumb = body.thumb as { size: number; classes: number[]; conf: number[] } | undefined;
  if (!thumb || !Array.isArray(thumb.classes) || !Array.isArray(thumb.conf)) {
    return NextResponse.json({ error: "A decision-boundary thumbnail is required" }, { status: 400 });
  }

  const name = String(body.name ?? "").trim().slice(0, 80) || "Untitled run";
  const author = String(body.author ?? "").trim().slice(0, 40) || "anonymous";

  const record: NewExperiment = {
    name,
    author,
    dataset: String(body.dataset ?? "spiral"),
    noise: num(body.noise),
    samples: Math.round(num(body.samples, 400)),
    hidden: Array.isArray(body.hidden) ? (body.hidden as number[]).map((h) => Math.round(num(h))) : [],
    features: Array.isArray(body.features) ? (body.features as string[]).map(String) : [],
    activation: String(body.activation ?? "tanh"),
    optimizer: String(body.optimizer ?? "adam"),
    learningRate: num(body.learningRate, 0.03),
    batchSize: Math.round(num(body.batchSize, 32)),
    l2: num(body.l2),
    epochs: Math.round(num(body.epochs)),
    paramCount: Math.round(num(body.paramCount)),
    trainLoss: num(body.trainLoss),
    testLoss: num(body.testLoss),
    trainAcc: num(body.trainAcc),
    testAcc: num(body.testAcc),
    history: Array.isArray(body.history)
      ? (body.history as { epoch: number; trainLoss: number; testLoss: number; testAcc: number }[])
      : [],
    thumb: {
      size: Math.round(num(thumb.size, 28)),
      classes: thumb.classes.map((c) => Math.round(num(c))),
      conf: thumb.conf.map((c) => Math.round(num(c) * 100) / 100),
    },
  };

  const [inserted] = await db.insert(experiments).values(record).returning();
  return NextResponse.json({ experiment: inserted }, { status: 201 });
}
