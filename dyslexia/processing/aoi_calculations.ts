/**
 * Area-of-interest measurement.
 *
 * The passage is rendered as one `<span data-word>` per word, so after layout we
 * can read each word's exact bounding box (CSS px relative to the passage
 * container). Line index is inferred by grouping words with the same vertical
 * offset. These boxes are returned to Python for fixation heatmaps.
 */

interface AreaOfInterest {
  word: string;
  line_index: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Split text into display words, keeping attached punctuation. */
export function splitIntoWords(text: string): string[] {
  return text
    .split(/\s+/)
    .map((word) => word.trim())
    .filter((word) => word.length > 0);
}

/** Measure rendered word boxes within a container, relative to its top-left. */
export function measureAreasOfInterest(container: HTMLElement): AreaOfInterest[] {
  const containerRect = container.getBoundingClientRect();
  const wordElements = Array.from(
    container.querySelectorAll<HTMLElement>("[data-word]")
  );

  // Group by rounded vertical offset to assign stable line indices.
  const lineTops: number[] = [];
  const lineIndexFor = (top: number): number => {
    const tolerance = 6;
    const existing = lineTops.findIndex((value) => Math.abs(value - top) <= tolerance);
    if (existing !== -1) {
      return existing;
    }
    lineTops.push(top);
    return lineTops.length - 1;
  };

  const areas = wordElements.map((element) => {
    const rect = element.getBoundingClientRect();
    const x = rect.left - containerRect.left;
    const y = rect.top - containerRect.top;
    return {
      word: element.dataset.word ?? element.textContent ?? "",
      line_index: lineIndexFor(Math.round(y)),
      x,
      y,
      width: rect.width,
      height: rect.height,
    } satisfies AreaOfInterest;
  });

  // Reassign line indices in top-to-bottom order for readability downstream.
  const sortedTops = [...lineTops].sort((a, b) => a - b);
  return areas.map((area) => ({
    ...area,
    line_index: sortedTops.indexOf(lineTops[area.line_index]),
  }));
}
