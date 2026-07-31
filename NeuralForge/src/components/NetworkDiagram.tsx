"use client";

import { useMemo } from "react";
import { cssColor } from "@/lib/nn/palette";

export interface LayerWeights {
  rows: number; // inputs
  cols: number; // outputs
  data: number[];
}

interface Props {
  inputLabels: string[];
  hidden: number[];
  outputs: number;
  weights: LayerWeights[];
}

/**
 * Live architecture view: every edge is a real weight from the model.
 * Blue = positive, orange = negative, thickness = |w|.
 */
export default function NetworkDiagram({ inputLabels, hidden, outputs, weights }: Props) {
  const width = 520;
  const height = 260;

  const { nodes, edges } = useMemo(() => {
    const sizes = [inputLabels.length, ...hidden, outputs];
    const colGap = width / (sizes.length + 0.6);
    const nodes: { x: number; y: number; layer: number; index: number }[] = [];

    sizes.forEach((count, layer) => {
      const x = colGap * (layer + 0.8);
      const usable = height - 36;
      for (let i = 0; i < count; i++) {
        const y = count === 1 ? height / 2 : 18 + (usable * i) / (count - 1);
        nodes.push({ x, y, layer, index: i });
      }
    });

    const nodeAt = (layer: number, index: number) =>
      nodes.find((n) => n.layer === layer && n.index === index)!;

    let maxAbs = 1e-6;
    for (const w of weights) for (const v of w.data) maxAbs = Math.max(maxAbs, Math.abs(v));

    const edges: { x1: number; y1: number; x2: number; y2: number; w: number }[] = [];
    weights.forEach((layerW, li) => {
      if (li + 1 >= sizes.length) return;
      const stride = layerW.rows * layerW.cols > 900 ? 3 : 1;
      for (let i = 0; i < layerW.rows; i += 1) {
        for (let o = 0; o < layerW.cols; o += 1) {
          if ((i * layerW.cols + o) % stride !== 0) continue;
          const a = nodeAt(li, i);
          const b = nodeAt(li + 1, o);
          if (!a || !b) continue;
          edges.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, w: layerW.data[i * layerW.cols + o] / maxAbs });
        }
      }
    });

    return { nodes, edges };
  }, [inputLabels.length, hidden, outputs, weights]);

  const lastLayer = hidden.length + 1;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full">
      {edges.map((e, i) => (
        <line
          key={i}
          x1={e.x1}
          y1={e.y1}
          x2={e.x2}
          y2={e.y2}
          stroke={e.w >= 0 ? "rgb(56,189,248)" : "rgb(251,146,60)"}
          strokeOpacity={Math.min(0.85, 0.08 + Math.abs(e.w) * 0.8)}
          strokeWidth={0.4 + Math.abs(e.w) * 2.4}
        />
      ))}
      {nodes.map((n, i) => {
        const isInput = n.layer === 0;
        const isOutput = n.layer === lastLayer;
        return (
          <g key={i}>
            <circle
              cx={n.x}
              cy={n.y}
              r={isInput || isOutput ? 8 : 6}
              fill={isOutput ? cssColor(n.index) : isInput ? "rgb(148,163,184)" : "rgb(30,41,59)"}
              stroke="rgba(226,232,240,0.55)"
              strokeWidth={1.2}
            />
            {isInput && (
              <text
                x={n.x - 14}
                y={n.y + 4}
                textAnchor="end"
                fontSize="10"
                fill="rgb(148,163,184)"
              >
                {inputLabels[n.index]}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
