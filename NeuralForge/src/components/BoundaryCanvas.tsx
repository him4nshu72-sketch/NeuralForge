"use client";

import { useEffect, useRef } from "react";
import type { Point } from "@/lib/nn/datasets";
import type { BoundaryGrid } from "@/lib/nn/trainer";
import { cssColor, paintBoundary } from "@/lib/nn/palette";

interface Props {
  boundary: BoundaryGrid | null;
  numClasses: number;
  trainPoints: Point[];
  testPoints: Point[];
  showTest: boolean;
  display?: number;
}

/**
 * Renders the model's decision surface as a low-res heatmap that is then
 * upscaled by the browser, with the dataset drawn on top.
 */
export default function BoundaryCanvas({
  boundary,
  numClasses,
  trainPoints,
  testPoints,
  showTest,
  display = 420,
}: Props) {
  const gridRef = useRef<HTMLCanvasElement>(null);
  const dotRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = gridRef.current;
    if (!canvas || !boundary) return;
    if (canvas.width !== boundary.size) {
      canvas.width = boundary.size;
      canvas.height = boundary.size;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    paintBoundary(ctx, boundary.size, boundary.classes, boundary.conf, numClasses);
  }, [boundary, numClasses]);

  useEffect(() => {
    const canvas = dotRef.current;
    if (!canvas) return;
    const dpr = typeof window === "undefined" ? 1 : Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = display * dpr;
    canvas.height = display * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, display, display);

    const toPx = (v: number) => ((v + 1) / 2) * display;
    const draw = (pts: Point[], radius: number, test: boolean) => {
      for (const p of pts) {
        const x = toPx(p.x);
        const y = display - toPx(p.y);
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = cssColor(p.label, test ? 0.35 : 1);
        ctx.fill();
        ctx.lineWidth = test ? 1.5 : 1;
        ctx.strokeStyle = test ? "rgba(255,255,255,0.9)" : "rgba(2,6,23,0.75)";
        ctx.stroke();
      }
    };

    draw(trainPoints, 3.4, false);
    if (showTest) draw(testPoints, 3.4, true);
  }, [trainPoints, testPoints, showTest, display]);

  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-white/10 bg-slate-950 shadow-2xl shadow-black/40"
      style={{ width: display, height: display, maxWidth: "100%" }}
    >
      <canvas
        ref={gridRef}
        className="absolute inset-0 h-full w-full"
        style={{ imageRendering: "auto" }}
      />
      <canvas ref={dotRef} className="absolute inset-0 h-full w-full" />
      <div className="pointer-events-none absolute inset-0 rounded-2xl ring-1 ring-inset ring-white/10" />
    </div>
  );
}
