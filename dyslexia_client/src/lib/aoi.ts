export interface AreaOfInterest {
  word: string
  line_index: number
  x: number
  y: number
  width: number
  height: number
}

// export function splitIntoWords(text: string): string[] {
//   return text.split(/\s+/).map(w => w.trim()).filter(w => w.length > 0)
// }

export function splitIntoWords(text: string): string[] {
  return text.match(/\S+|\n/g) ?? []
}

export function measureAreasOfInterest(container: HTMLElement): AreaOfInterest[] {
  const containerRect = container.getBoundingClientRect()
  const wordElements = Array.from(container.querySelectorAll<HTMLElement>('[data-word]'))

  const lineTops: number[] = []
  const lineIndexFor = (top: number): number => {
    const tolerance = 6
    const existing = lineTops.findIndex(v => Math.abs(v - top) <= tolerance)
    if (existing !== -1) return existing
    lineTops.push(top)
    return lineTops.length - 1
  }

  const areas = wordElements.map(el => {
    const rect = el.getBoundingClientRect()
    const x = rect.left - containerRect.left
    const y = rect.top  - containerRect.top
    return {
      word: el.dataset.word ?? el.textContent ?? '',
      line_index: lineIndexFor(Math.round(y)),
      x, y,
      width:  rect.width,
      height: rect.height,
    } satisfies AreaOfInterest
  })

  const sortedTops = [...lineTops].sort((a, b) => a - b)
  return areas.map(area => ({
    ...area,
    line_index: sortedTops.indexOf(lineTops[area.line_index]),
  }))
}
