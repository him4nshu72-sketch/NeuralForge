import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuralForge — build & train neural networks from scratch",
  description:
    "An interactive deep-learning lab: a neural network written from scratch in TypeScript, trained live in your browser, with a Postgres-backed experiment leaderboard.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_20%_-10%,rgba(56,189,248,0.16),transparent_55%),radial-gradient(circle_at_85%_10%,rgba(244,114,182,0.12),transparent_50%)]" />
        <header className="sticky top-0 z-30 border-b border-white/10 bg-slate-950/80 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] items-center gap-6 px-4 py-3">
            <Link href="/" className="flex items-center gap-2.5">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 text-sm font-black text-slate-950">
                N
              </span>
              <span className="text-sm font-bold tracking-tight">
                Neural<span className="text-cyan-400">Forge</span>
              </span>
            </Link>
            <nav className="flex items-center gap-1 text-sm">
              <Link
                href="/"
                className="rounded-lg px-3 py-1.5 text-slate-400 transition hover:bg-white/5 hover:text-white"
              >
                Lab
              </Link>
              <Link
                href="/leaderboard"
                className="rounded-lg px-3 py-1.5 text-slate-400 transition hover:bg-white/5 hover:text-white"
              >
                Leaderboard
              </Link>
              <Link
                href="/learn"
                className="rounded-lg px-3 py-1.5 text-slate-400 transition hover:bg-white/5 hover:text-white"
              >
                How it works
              </Link>
            </nav>
            <span className="ml-auto hidden rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] text-slate-400 sm:block">
              zero ML libraries · pure TypeScript backprop
            </span>
          </div>
        </header>
        <main className="mx-auto max-w-[1400px] px-4 py-6">{children}</main>
        <footer className="mx-auto max-w-[1400px] px-4 pb-10 pt-4 text-xs text-slate-600">
          Built with Next.js, Drizzle ORM and PostgreSQL. The neural network — matrices,
          backpropagation, SGD/Momentum/Adam — is implemented by hand in{" "}
          <code className="text-slate-500">src/lib/nn/engine.ts</code>.
        </footer>
      </body>
    </html>
  );
}
