"use client";

import { useEffect, useRef } from "react";
import { paintBoundary } from "@/lib/nn/palette";

export default function Thumb({
  thumb,
  numClasses,
  size = 56,
}: {
  thumb: { size: number; classes: number[]; conf: number[] };
  numClasses: number;
  size?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = thumb.size;
    canvas.height = thumb.size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    paintBoundary(
      ctx,
      thumb.size,
      Uint8Array.from(thumb.classes),
      Float32Array.from(thumb.conf),
      Math.max(2, numClasses),
    );
  }, [thumb, numClasses]);

  return (
    <canvas
      ref={ref}
      className="rounded-lg border border-white/10"
      style={{ width: size, height: size }}
    />
  );
}
