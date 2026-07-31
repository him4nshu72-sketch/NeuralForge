/** Class colours shared by the canvas renderers and the legends. */
export const CLASS_COLORS: [number, number, number][] = [
  [56, 189, 248], // sky
  [244, 114, 182], // pink
  [163, 230, 53], // lime
  [251, 191, 36], // amber
  [167, 139, 250], // violet
];

export function cssColor(index: number, alpha = 1): string {
  const [r, g, b] = CLASS_COLORS[index % CLASS_COLORS.length];
  return alpha >= 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Paint a decision-boundary grid into an ImageData buffer.
 * Confidence controls how strongly the class colour is blended with the
 * dark background, which makes uncertain regions fade out.
 */
export function paintBoundary(
  ctx: CanvasRenderingContext2D,
  size: number,
  classes: Uint8Array,
  conf: Float32Array,
  numClasses: number,
): void {
  const image = ctx.createImageData(size, size);
  const data = image.data;
  const floor = 1 / numClasses;
  for (let i = 0; i < classes.length; i++) {
    const [r, g, b] = CLASS_COLORS[classes[i] % CLASS_COLORS.length];
    // rescale confidence from [1/k, 1] -> [0, 1]
    const t = Math.max(0, Math.min(1, (conf[i] - floor) / (1 - floor)));
    const a = 0.12 + 0.68 * t;
    const o = i * 4;
    data[o] = Math.round(r * a + 9 * (1 - a));
    data[o + 1] = Math.round(g * a + 12 * (1 - a));
    data[o + 2] = Math.round(b * a + 20 * (1 - a));
    data[o + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}
