"use client";

import { useState } from "react";
import { runAllGradientChecks, type GradCheckResult } from "@/lib/nn/gradcheck";

export default function GradCheck() {
  const [results, setResults] = useState<GradCheckResult[] | null>(null);
  const [busy, setBusy] = useState(false);

  const run = () => {
    setBusy(true);
    // let the browser paint the "running" state first
    setTimeout(() => {
      setResults(runAllGradientChecks());
      setBusy(false);
    }, 30);
  };

  const allPassed = results?.every((r) => r.passed);

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
      <h2 className="text-base font-bold">Proof: numerical gradient check</h2>
      <p className="mt-2 text-sm leading-relaxed text-slate-400">
        How do you know hand-derived gradients are right? You compare them against a numerical
        estimate of the same derivative. Press the button to run the check live in your browser
        for all four activation functions.
      </p>

      <button
        type="button"
        onClick={run}
        disabled={busy}
        className="mt-3 rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
      >
        {busy ? "Running…" : "Run gradient check"}
      </button>

      {results && (
        <div className="mt-4 space-y-2">
          <div
            className={`rounded-lg px-3 py-2 text-sm font-semibold ${
              allPassed ? "bg-lime-400/15 text-lime-300" : "bg-rose-400/15 text-rose-300"
            }`}
          >
            {allPassed
              ? "✓ All gradients match finite differences"
              : "✗ Mismatch detected"}
          </div>
          <table className="w-full text-left text-xs">
            <thead className="text-[10px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="py-1.5">Activation</th>
                <th className="py-1.5">Weights checked</th>
                <th className="py-1.5">Max rel. error</th>
                <th className="py-1.5">Mean rel. error</th>
                <th className="py-1.5" />
              </tr>
            </thead>
            <tbody className="font-mono text-slate-300">
              {results.map((r) => (
                <tr key={r.activation} className="border-t border-white/5">
                  <td className="py-1.5">{r.activation}</td>
                  <td className="py-1.5">{r.samples}</td>
                  <td className="py-1.5">{r.maxRelativeError.toExponential(2)}</td>
                  <td className="py-1.5">{r.meanRelativeError.toExponential(2)}</td>
                  <td className="py-1.5">{r.passed ? "✅" : "❌"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
