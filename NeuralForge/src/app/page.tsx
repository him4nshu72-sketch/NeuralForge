import Lab from "@/components/Lab";

export default function HomePage() {
  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-white/10 bg-gradient-to-br from-white/[0.07] to-transparent p-6 sm:p-8">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">
          Interactive deep-learning lab
        </p>
        <h1 className="mt-2 max-w-3xl text-3xl font-black leading-tight tracking-tight sm:text-4xl">
          Design a neural network, watch it learn, and see the exact moment it
          figures out the pattern.
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-400">
          Every neuron, gradient and optimizer step below is computed by an engine written
          from scratch in TypeScript — no TensorFlow, no PyTorch. Change the architecture,
          hit <span className="text-slate-200">Train</span>, and the decision boundary
          re-renders 60 times a second. Save your best runs to the leaderboard.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-slate-400">
          {[
            "hand-derived backpropagation",
            "SGD · Momentum · Adam",
            "softmax + cross-entropy",
            "He / Xavier init",
            "L2 regularisation",
            "feature engineering",
          ].map((t) => (
            <span key={t} className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
              {t}
            </span>
          ))}
        </div>
      </section>

      <Lab />
    </div>
  );
}
