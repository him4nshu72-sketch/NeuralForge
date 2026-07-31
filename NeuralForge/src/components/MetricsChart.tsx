"use client";

import { useMemo } from "react";

export interface Series {
  label: string;
  color: string;
  values: number[];
}

interface Props {
  series: Series[];
  height?: number;
  yMax?: number;
  yLabel?: string;
}

export default function MetricsChart({ series, height = 140, yMax, yLabel }: Props) {
  const width = 520;

  const { paths, top } = useMemo(() => {
    const maxLen = Math.max(1, ...series.map((s) => s.values.length));
    const dataMax = Math.max(
      1e-6,
      ...series.flatMap((s) => (s.values.length ? s.values : [0])),
    );
    const top = yMax ?? Math.max(dataMax * 1.15, 0.1);

    const paths = series.map((s) => {
      if (s.values.length < 2) return { ...s, d: "" };
      const step = width / (maxLen - 1);
      const d = s.values
        .map((v, i) => {
          const x = i * step;
          const y = height - Math.min(1, v / top) * (height - 6) - 3;
          return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ");
      return { ...s, d };
    });
    return { paths, top };
  }, [series, height, yMax]);

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-[140px] w-full rounded-xl border border-white/10 bg-slate-950/60"
      >
        {[0.25, 0.5, 0.75].map((f) => (
          <line
            key={f}
            x1={0}
            x2={width}
            y1={height * f}
            y2={height * f}
            stroke="rgba(148,163,184,0.14)"
            strokeWidth={1}
          />
        ))}
        {paths.map((p) => (
          <path
            key={p.label}
            d={p.d}
            fill="none"
            stroke={p.color}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-400">
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <span className="h-[3px] w-4 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        <span className="ml-auto tabular-nums text-slate-500">
          {yLabel ?? "max"} {top.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
