"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import BoundaryCanvas from "./BoundaryCanvas";
import MetricsChart from "./MetricsChart";
import NetworkDiagram, { type LayerWeights } from "./NetworkDiagram";
import { Panel, Segmented, Slider, Stat } from "./ui";
import {
  DATASETS,
  FEATURES,
  featureLabel,
  type DatasetName,
  type FeatureId,
} from "@/lib/nn/datasets";
import { ACTIVATIONS, OPTIMIZERS, type Activation, type OptimizerName } from "@/lib/nn/engine";
import { cssColor } from "@/lib/nn/palette";
import {
  DEFAULT_CONFIG,
  Trainer,
  type BoundaryGrid,
  type Metrics,
  type TrainConfig,
} from "@/lib/nn/trainer";

const GRID_RESOLUTION = 72;
const THUMB_RESOLUTION = 26;

const LR_OPTIONS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
const BATCH_OPTIONS = [8, 16, 32, 64, 128];
const L2_OPTIONS = [0, 0.0001, 0.001, 0.01];

export default function Lab() {
  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [showTest, setShowTest] = useState(true);

  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState<Metrics[]>([]);
  const [boundary, setBoundary] = useState<BoundaryGrid | null>(null);
  const [weights, setWeights] = useState<LayerWeights[]>([]);
  const [tick, setTick] = useState(0);

  const trainerRef = useRef<Trainer | null>(null);
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(running);
  const speedRef = useRef(speed);
  runningRef.current = running;
  speedRef.current = speed;

  const configKey = JSON.stringify(config);

  const snapshot = useCallback((trainer: Trainer) => {
    setMetrics(trainer.latest);
    setHistory(trainer.compactHistory(160));
    setBoundary(trainer.boundary(GRID_RESOLUTION));
    setWeights(
      trainer.model.layers.map((_, i) => {
        const w = trainer.model.weightsOf(i);
        return { rows: w.rows, cols: w.cols, data: Array.from(w.data) };
      }),
    );
    setTick((t) => t + 1);
  }, []);

  // (Re)build the model whenever any hyper-parameter changes.
  useEffect(() => {
    const parsed = JSON.parse(configKey) as TrainConfig;
    const trainer = new Trainer(parsed);
    trainerRef.current = trainer;
    snapshot(trainer);
  }, [configKey, snapshot]);

  // Animation loop: train N epochs per frame, then repaint.
  useEffect(() => {
    if (!running) return;
    let frame = 0;
    const loop = () => {
      const trainer = trainerRef.current;
      if (!trainer || !runningRef.current) return;
      for (let i = 0; i < speedRef.current; i++) trainer.runEpoch();
      trainer.record();
      frame++;
      if (frame % 2 === 0) snapshot(trainer);
      else setMetrics(trainer.latest);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [running, snapshot]);

  const patch = (p: Partial<TrainConfig>) => setConfig((c) => ({ ...c, ...p }));

  const stepOnce = () => {
    const trainer = trainerRef.current;
    if (!trainer) return;
    trainer.runEpoch();
    trainer.record();
    snapshot(trainer);
  };

  const reset = () => {
    setRunning(false);
    const trainer = new Trainer(JSON.parse(configKey) as TrainConfig);
    trainerRef.current = trainer;
    snapshot(trainer);
  };

  const toggleFeature = (id: FeatureId) => {
    setConfig((c) => {
      const has = c.features.includes(id);
      if (has && c.features.length === 1) return c;
      return {
        ...c,
        features: has ? c.features.filter((f) => f !== id) : [...c.features, id],
      };
    });
  };

  const changeLayerSize = (index: number, delta: number) => {
    setConfig((c) => {
      const hidden = c.hidden.slice();
      hidden[index] = Math.max(1, Math.min(16, hidden[index] + delta));
      return { ...c, hidden };
    });
  };

  const addLayer = () =>
    setConfig((c) => (c.hidden.length >= 5 ? c : { ...c, hidden: [...c.hidden, 4] }));
  const removeLayer = () =>
    setConfig((c) => (c.hidden.length === 0 ? c : { ...c, hidden: c.hidden.slice(0, -1) }));

  const trainer = trainerRef.current;
  const numClasses = trainer?.classes ?? 2;
  const paramCount = trainer?.paramCount ?? 0;
  const gap = metrics ? metrics.trainAcc - metrics.testAcc : 0;

  const lossSeries = useMemo(
    () => [
      { label: "train loss", color: "rgb(56,189,248)", values: history.map((h) => h.trainLoss) },
      { label: "test loss", color: "rgb(244,114,182)", values: history.map((h) => h.testLoss) },
    ],
    [history],
  );
  const accSeries = useMemo(
    () => [
      { label: "train acc", color: "rgb(163,230,53)", values: history.map((h) => h.trainAcc) },
      { label: "test acc", color: "rgb(251,191,36)", values: history.map((h) => h.testAcc) },
    ],
    [history],
  );

  return (
    <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)_340px]">
      {/* ---------------- left: data + architecture ---------------- */}
      <div className="space-y-4">
        <Panel title="Dataset" subtitle={DATASETS.find((d) => d.id === config.dataset)?.blurb}>
          <div className="grid grid-cols-3 gap-1.5">
            {DATASETS.map((d) => (
              <button
                key={d.id}
                type="button"
                onClick={() => patch({ dataset: d.id as DatasetName })}
                className={`rounded-lg border px-2 py-2 text-xs font-medium transition ${
                  config.dataset === d.id
                    ? "border-cyan-400/60 bg-cyan-400/15 text-cyan-200"
                    : "border-white/10 bg-slate-900/60 text-slate-400 hover:border-white/25 hover:text-slate-200"
                }`}
              >
                {d.label}
              </button>
            ))}
          </div>
          <div className="mt-4 space-y-3">
            <Slider
              label="Samples"
              min={100}
              max={1000}
              step={50}
              value={config.samples}
              onChange={(v) => patch({ samples: v })}
            />
            <Slider
              label="Noise"
              min={0}
              max={0.3}
              step={0.01}
              value={config.noise}
              format={(v) => v.toFixed(2)}
              onChange={(v) => patch({ noise: v })}
            />
            <Slider
              label="Train / test split"
              min={0.3}
              max={0.9}
              step={0.05}
              value={config.trainRatio}
              format={(v) => `${Math.round(v * 100)}% train`}
              onChange={(v) => patch({ trainRatio: v })}
            />
          </div>
          <button
            type="button"
            onClick={() => patch({ seed: Math.floor(Math.random() * 100000) })}
            className="mt-4 w-full rounded-lg border border-white/10 bg-slate-900/70 px-3 py-2 text-xs font-medium text-slate-300 hover:border-white/25 hover:text-white"
          >
            🎲 Resample data (seed {config.seed})
          </button>
        </Panel>

        <Panel title="Input features" subtitle="Feature engineering beats depth. Try x₁² + x₂².">
          <div className="grid grid-cols-4 gap-1.5">
            {FEATURES.map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => toggleFeature(f.id)}
                className={`rounded-lg border px-1 py-2 text-[11px] transition ${
                  config.features.includes(f.id)
                    ? "border-lime-400/50 bg-lime-400/15 text-lime-200"
                    : "border-white/10 bg-slate-900/60 text-slate-500 hover:text-slate-300"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </Panel>

        <Panel
          title="Architecture"
          right={
            <span className="rounded-full bg-slate-900 px-2 py-0.5 font-mono text-[10px] text-slate-400">
              {paramCount} params
            </span>
          }
        >
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={removeLayer}
              className="h-7 w-7 rounded-lg border border-white/10 bg-slate-900/70 text-slate-300 hover:border-white/30"
            >
              −
            </button>
            <span className="text-xs text-slate-400">
              {config.hidden.length} hidden layer{config.hidden.length === 1 ? "" : "s"}
            </span>
            <button
              type="button"
              onClick={addLayer}
              className="h-7 w-7 rounded-lg border border-white/10 bg-slate-900/70 text-slate-300 hover:border-white/30"
            >
              +
            </button>
          </div>
          <div className="mt-3 space-y-2">
            {config.hidden.map((units, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-12 text-[11px] text-slate-500">L{i + 1}</span>
                <button
                  type="button"
                  onClick={() => changeLayerSize(i, -1)}
                  className="h-6 w-6 rounded border border-white/10 bg-slate-900/70 text-xs text-slate-300"
                >
                  −
                </button>
                <div className="flex flex-1 gap-[3px]">
                  {Array.from({ length: units }).map((_, k) => (
                    <span key={k} className="h-5 flex-1 rounded-sm bg-cyan-400/70" />
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => changeLayerSize(i, 1)}
                  className="h-6 w-6 rounded border border-white/10 bg-slate-900/70 text-xs text-slate-300"
                >
                  +
                </button>
                <span className="w-5 text-right font-mono text-[11px] text-slate-400">{units}</span>
              </div>
            ))}
            {config.hidden.length === 0 && (
              <p className="text-[11px] text-amber-300/80">
                No hidden layers → this is plain logistic regression. It can only draw straight lines.
              </p>
            )}
          </div>
        </Panel>
      </div>

      {/* ---------------- center: the visualisation ---------------- */}
      <div className="space-y-4">
        <Panel>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setRunning((r) => !r)}
              className={`rounded-xl px-5 py-2.5 text-sm font-semibold transition ${
                running
                  ? "bg-amber-400 text-slate-950 hover:bg-amber-300"
                  : "bg-cyan-400 text-slate-950 hover:bg-cyan-300"
              }`}
            >
              {running ? "❚❚ Pause" : "▶ Train"}
            </button>
            <button
              type="button"
              onClick={stepOnce}
              className="rounded-xl border border-white/10 bg-slate-900/70 px-4 py-2.5 text-sm text-slate-200 hover:border-white/30"
            >
              Step
            </button>
            <button
              type="button"
              onClick={reset}
              className="rounded-xl border border-white/10 bg-slate-900/70 px-4 py-2.5 text-sm text-slate-200 hover:border-white/30"
            >
              ↺ Reset
            </button>
            <div className="ml-auto flex items-center gap-2 text-xs text-slate-400">
              <span>speed</span>
              <Segmented
                value={String(speed)}
                onChange={(v) => setSpeed(Number(v))}
                options={[
                  { value: "1", label: "1×" },
                  { value: "2", label: "2×" },
                  { value: "5", label: "5×" },
                  { value: "10", label: "10×" },
                ]}
              />
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="Epoch" value={String(metrics?.epoch ?? 0)} />
            <Stat label="Train loss" value={(metrics?.trainLoss ?? 0).toFixed(4)} />
            <Stat
              label="Test acc"
              value={`${((metrics?.testAcc ?? 0) * 100).toFixed(1)}%`}
              tone={(metrics?.testAcc ?? 0) > 0.9 ? "good" : "default"}
            />
            <Stat
              label="Overfit gap"
              value={`${(gap * 100).toFixed(1)}%`}
              tone={gap > 0.12 ? "warn" : "default"}
            />
          </div>
        </Panel>

        <Panel
          title="Decision boundary"
          subtitle="Every pixel is a forward pass through your network."
          right={
            <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <input
                type="checkbox"
                checked={showTest}
                onChange={(e) => setShowTest(e.target.checked)}
                className="accent-cyan-400"
              />
              show test set
            </label>
          }
        >
          <div className="flex justify-center">
            <BoundaryCanvas
              boundary={boundary}
              numClasses={numClasses}
              trainPoints={trainer?.trainPoints ?? []}
              testPoints={trainer?.testPoints ?? []}
              showTest={showTest}
              display={440}
            />
          </div>
          <div className="mt-3 flex flex-wrap justify-center gap-3 text-[11px] text-slate-400">
            {Array.from({ length: numClasses }).map((_, c) => (
              <span key={c} className="inline-flex items-center gap-1.5">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ background: cssColor(c) }}
                />
                class {c}
              </span>
            ))}
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full border border-white bg-white/30" />
              test point
            </span>
          </div>
        </Panel>

        <div className="grid gap-4 md:grid-cols-2">
          <Panel title="Loss" subtitle="cross-entropy">
            <MetricsChart series={lossSeries} />
          </Panel>
          <Panel title="Accuracy">
            <MetricsChart series={accSeries} yMax={1} yLabel="max" />
          </Panel>
        </div>
      </div>

      {/* ---------------- right: optimisation + save ---------------- */}
      <div className="space-y-4">
        <Panel title="Optimisation">
          <div className="space-y-3">
            <div>
              <p className="mb-1.5 text-xs text-slate-400">Activation</p>
              <Segmented
                value={config.activation}
                onChange={(v) => patch({ activation: v as Activation })}
                options={ACTIVATIONS.map((a) => ({ value: a, label: a === "leakyRelu" ? "leaky" : a }))}
              />
            </div>
            <div>
              <p className="mb-1.5 text-xs text-slate-400">Optimizer</p>
              <Segmented
                value={config.optimizer}
                onChange={(v) => patch({ optimizer: v as OptimizerName })}
                options={OPTIMIZERS.map((o) => ({ value: o, label: o }))}
              />
            </div>
            <label className="block text-xs text-slate-400">
              Learning rate
              <select
                value={config.learningRate}
                onChange={(e) => patch({ learningRate: Number(e.target.value) })}
                className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-2 py-1.5 text-sm text-slate-200"
              >
                {LR_OPTIONS.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="block text-xs text-slate-400">
                Batch size
                <select
                  value={config.batchSize}
                  onChange={(e) => patch({ batchSize: Number(e.target.value) })}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-2 py-1.5 text-sm text-slate-200"
                >
                  {BATCH_OPTIONS.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-xs text-slate-400">
                L2 (weight decay)
                <select
                  value={config.l2}
                  onChange={(e) => patch({ l2: Number(e.target.value) })}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-2 py-1.5 text-sm text-slate-200"
                >
                  {L2_OPTIONS.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
        </Panel>

        <Panel title="Live network" subtitle="blue = positive weight · orange = negative">
          <NetworkDiagram
            key={`${config.features.length}-${config.hidden.join(",")}-${numClasses}-${tick % 2}`}
            inputLabels={config.features.map(featureLabel)}
            hidden={config.hidden}
            outputs={numClasses}
            weights={weights}
          />
        </Panel>

        <SavePanel config={config} metrics={metrics} trainerRef={trainerRef} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function SavePanel({
  config,
  metrics,
  trainerRef,
}: {
  config: TrainConfig;
  metrics: Metrics | null;
  trainerRef: React.RefObject<Trainer | null>;
}) {
  const [name, setName] = useState("");
  const [author, setAuthor] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [message, setMessage] = useState("");

  const save = async () => {
    const trainer = trainerRef.current;
    if (!trainer || !metrics) return;
    if (metrics.epoch === 0) {
      setStatus("error");
      setMessage("Train the network for at least one epoch first.");
      return;
    }
    setStatus("saving");
    const thumbGrid = trainer.boundary(THUMB_RESOLUTION);
    const payload = {
      name: name || `${config.dataset} · ${config.hidden.join("-") || "linear"}`,
      author,
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
      epochs: metrics.epoch,
      paramCount: trainer.paramCount,
      trainLoss: metrics.trainLoss,
      testLoss: metrics.testLoss,
      trainAcc: metrics.trainAcc,
      testAcc: metrics.testAcc,
      history: trainer.compactHistory(80),
      thumb: {
        size: thumbGrid.size,
        classes: Array.from(thumbGrid.classes),
        conf: Array.from(thumbGrid.conf),
      },
    };

    try {
      const res = await fetch("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).error ?? "Save failed");
      setStatus("saved");
      setMessage("Saved! It's on the leaderboard now.");
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Save failed");
    }
  };

  return (
    <Panel title="Save experiment" subtitle="Ship your run to the public leaderboard.">
      <div className="space-y-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Run name (e.g. deep tanh spiral)"
          className="w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600"
        />
        <input
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          placeholder="Your handle"
          className="w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600"
        />
        <button
          type="button"
          onClick={save}
          disabled={status === "saving"}
          className="w-full rounded-lg bg-lime-400 px-3 py-2 text-sm font-semibold text-slate-950 transition hover:bg-lime-300 disabled:opacity-50"
        >
          {status === "saving" ? "Saving…" : "💾 Save run"}
        </button>
        {message && (
          <p className={`text-xs ${status === "error" ? "text-rose-300" : "text-lime-300"}`}>
            {message}{" "}
            {status === "saved" && (
              <Link href="/leaderboard" className="underline">
                View leaderboard →
              </Link>
            )}
          </p>
        )}
      </div>
    </Panel>
  );
}
