import Link from "next/link";
import { desc } from "drizzle-orm";
import { db } from "@/db";
import { experiments } from "@/db/schema";
import { classesOf, featureLabel, type DatasetName } from "@/lib/nn/datasets";
import Thumb from "@/components/Thumb";
import DeleteRunButton from "@/components/DeleteRunButton";

export const dynamic = "force-dynamic";

export default async function LeaderboardPage({
  searchParams,
}: {
  searchParams: Promise<{ sort?: string }>;
}) {
  const { sort } = await searchParams;
  const byRecent = sort === "recent";

  let rows: (typeof experiments.$inferSelect)[] = [];
  let dbError = false;
  try {
    rows = await db
      .select()
      .from(experiments)
      .orderBy(byRecent ? desc(experiments.createdAt) : desc(experiments.testAcc))
      .limit(100);
  } catch {
    dbError = true;
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black tracking-tight">Experiment leaderboard</h1>
          <p className="mt-1 text-sm text-slate-400">
            Every saved run, ranked by test-set accuracy. The thumbnail is the actual
            decision surface the model learned.
          </p>
        </div>
        <div className="flex gap-1 rounded-lg bg-slate-900/70 p-1 text-xs">
          <Link
            href="/leaderboard"
            className={`rounded-md px-3 py-1.5 ${!byRecent ? "bg-cyan-400 text-slate-950" : "text-slate-400 hover:text-white"}`}
          >
            Top accuracy
          </Link>
          <Link
            href="/leaderboard?sort=recent"
            className={`rounded-md px-3 py-1.5 ${byRecent ? "bg-cyan-400 text-slate-950" : "text-slate-400 hover:text-white"}`}
          >
            Most recent
          </Link>
        </div>
      </header>

      {dbError && (
        <p className="rounded-xl border border-amber-400/30 bg-amber-400/10 p-4 text-sm text-amber-200">
          Could not reach the database. Run <code>npx drizzle-kit push</code> to create the
          tables.
        </p>
      )}

      {!dbError && rows.length === 0 && (
        <div className="rounded-2xl border border-dashed border-white/15 bg-white/[0.02] p-10 text-center">
          <p className="text-slate-300">No experiments yet.</p>
          <Link
            href="/"
            className="mt-3 inline-block rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950"
          >
            Train the first model →
          </Link>
        </div>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-2xl border border-white/10 bg-white/[0.03]">
          <table className="w-full min-w-[860px] text-sm">
            <thead>
              <tr className="border-b border-white/10 text-left text-[11px] uppercase tracking-wider text-slate-500">
                <th className="px-4 py-3">#</th>
                <th className="px-2 py-3">Boundary</th>
                <th className="px-3 py-3">Run</th>
                <th className="px-3 py-3">Dataset</th>
                <th className="px-3 py-3">Architecture</th>
                <th className="px-3 py-3">Optimizer</th>
                <th className="px-3 py-3 text-right">Epochs</th>
                <th className="px-3 py-3 text-right">Train acc</th>
                <th className="px-3 py-3 text-right">Test acc</th>
                <th className="px-3 py-3" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => {
                const medal = ["🥇", "🥈", "🥉"][i];
                return (
                  <tr
                    key={row.id}
                    className="border-b border-white/5 transition last:border-0 hover:bg-white/[0.04]"
                  >
                    <td className="px-4 py-3 font-mono text-slate-500">
                      {byRecent ? i + 1 : (medal ?? i + 1)}
                    </td>
                    <td className="px-2 py-2">
                      <Thumb
                        thumb={row.thumb}
                        numClasses={classesOf(row.dataset as DatasetName)}
                      />
                    </td>
                    <td className="px-3 py-3">
                      <div className="font-medium text-slate-100">{row.name}</div>
                      <div className="text-[11px] text-slate-500">
                        @{row.author} ·{" "}
                        {new Date(row.createdAt).toLocaleDateString(undefined, {
                          month: "short",
                          day: "numeric",
                        })}
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      <div className="text-slate-300">{row.dataset}</div>
                      <div className="text-[11px] text-slate-500">
                        noise {row.noise.toFixed(2)} · n={row.samples}
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      <div className="font-mono text-[12px] text-cyan-300">
                        {row.features.length}→{row.hidden.join("→") || "·"}→out
                      </div>
                      <div className="text-[11px] text-slate-500">
                        {row.activation} · {row.paramCount} params ·{" "}
                        {row.features.map(featureLabel).join(" ")}
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      <div className="text-slate-300">{row.optimizer}</div>
                      <div className="text-[11px] text-slate-500">
                        lr {row.learningRate} · bs {row.batchSize}
                        {row.l2 > 0 ? ` · L2 ${row.l2}` : ""}
                      </div>
                    </td>
                    <td className="px-3 py-3 text-right font-mono text-slate-400">
                      {row.epochs}
                    </td>
                    <td className="px-3 py-3 text-right font-mono text-slate-400">
                      {(row.trainAcc * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-3 text-right">
                      <span
                        className={`font-mono text-base ${
                          row.testAcc >= 0.95
                            ? "text-lime-300"
                            : row.testAcc >= 0.85
                              ? "text-cyan-300"
                              : "text-slate-300"
                        }`}
                      >
                        {(row.testAcc * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-3 py-3 text-right">
                      <DeleteRunButton id={row.id} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
