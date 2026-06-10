<template>
  <div class="text-left bg-white rounded-xl p-5">
    <h3 class="font-semibold mb-4 text-center">Heatmap de lecture</h3>

    <div
      ref="textContainer"
      class="relative text-lg leading-loose max-w-3xl mx-auto select-none"
      style="word-break: normal; overflow-wrap: break-word; word-wrap: break-word;"
    >
      <template v-for="(token, i) in words" :key="i">
        <br v-if="token === '\n'" />
        <span v-else :data-word="token" class="inline-block">{{ token }}&nbsp;</span>
      </template>

      <canvas
        ref="canvasRef"
        class="absolute top-0 left-0 pointer-events-none rounded-xl"
        :width="canvasWidth"
        :height="canvasHeight"
        :style="{ width: canvasWidth + 'px', height: canvasHeight + 'px' }"
      />
    </div>

    <p class="text-center text-sm text-gray-400 mt-2">
      {{ gazePoints.length }} points — {{ aois.length }} mots analysés
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { splitIntoWords, measureAreasOfInterest } from '@/lib/aoi'
import type { AreaOfInterest } from '@/lib/aoi'
import type { GazePoint } from '@/types/index'

const props = defineProps<{
  text: string
  gazePoints: GazePoint[]
}>()

const textContainer = ref<HTMLElement | null>(null)
const canvasRef     = ref<HTMLCanvasElement | null>(null)
const canvasWidth   = ref(800)
const canvasHeight  = ref(300)
const words = splitIntoWords(props.text)
const aois  = ref<AreaOfInterest[]>([])

let resizeObserver: ResizeObserver | null = null

interface WordHit { aoi: AreaOfInterest; count: number }

// ── Conversion angles → position canvas ──────────────────────────────────

function anglesToCanvas(gazePoints: GazePoint[]): { px: number; py: number }[] {
  if (gazePoints.length === 0) return []

  const pairs = gazePoints.map(p => ({
    yaw:   (p.yaw_l + p.yaw_r) / 2,
    pitch: (p.pitch_l + p.pitch_r) / 2,
  }))

  const yaws    = pairs.map(p => p.yaw)
  const pitches = pairs.map(p => p.pitch)
  const minYaw  = Math.min(...yaws),    maxYaw   = Math.max(...yaws)
  const minPitch = Math.min(...pitches), maxPitch = Math.max(...pitches)
  const yawRange   = maxYaw   - minYaw   || 1
  const pitchRange = maxPitch - minPitch || 1

  const padX = canvasWidth.value  * 0.05
  const padY = canvasHeight.value * 0.05
  const w    = canvasWidth.value  - padX * 2
  const h    = canvasHeight.value - padY * 2

  return pairs.map(p => ({
    px: padX + ((p.yaw   - minYaw)   / yawRange)   * w,
    py: padY + ((p.pitch - minPitch) / pitchRange) * h,
  }))
}

// ── Fixations par mot ─────────────────────────────────────────────────────

function computeHits(aoiList: AreaOfInterest[]): WordHit[] {
  const hits: WordHit[] = aoiList.map(aoi => ({ aoi, count: 0 }))
  const coords = anglesToCanvas(props.gazePoints)

  for (const { px, py } of coords) {
    for (const hit of hits) {
      if (
        px >= hit.aoi.x &&
        px <= hit.aoi.x + hit.aoi.width &&
        py >= hit.aoi.y &&
        py <= hit.aoi.y + hit.aoi.height
      ) {
        hit.count++
      }
    }
  }
  return hits
}

// ── Dessin ────────────────────────────────────────────────────────────────

function drawHeatmap(hits: WordHit[]) {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const maxCount = Math.max(...hits.map(h => h.count), 1)

  for (const hit of hits) {
    if (hit.count === 0) continue
    // const intensity = hit.count / maxCount
    const intensity = hit.count / 200
    const hue   = (1 - intensity) * 240
    const alpha = 0.15 + intensity * 0.7

    ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${alpha})`
    ctx.beginPath()
    ctx.roundRect(hit.aoi.x - 2, hit.aoi.y - 2, hit.aoi.width + 4, hit.aoi.height + 4, 4)
    ctx.fill()

    ctx.fillStyle = `hsla(${hue}, 100%, 20%, ${Math.min(alpha + 0.3, 1)})`
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(String(hit.count), hit.aoi.x + hit.aoi.width / 2, hit.aoi.y - 4)
  }
}

// ── Init : attend que le layout soit stable ───────────────────────────────

async function init() {
  await nextTick()
  const container = textContainer.value
  if (!container) return

  // Mesure après que le texte soit wrappé
  const rect = container.getBoundingClientRect()
  canvasWidth.value  = Math.round(rect.width)
  canvasHeight.value = Math.round(rect.height)

  // Attendre que le canvas se redimensionne
  await nextTick()

  // measureAreasOfInterest retourne des coords relatives au container
  // Le padding CSS est déjà inclus dans getBoundingClientRect des spans
  aois.value = measureAreasOfInterest(container)
  drawHeatmap(computeHits(aois.value))
}

onMounted(async () => {
  await nextTick()
  const container = textContainer.value
  if (!container) return

  // ResizeObserver pour re-dessiner si la fenêtre change
  resizeObserver = new ResizeObserver(() => init())
  resizeObserver.observe(container)

  await init()
})

onUnmounted(() => {
  resizeObserver?.disconnect()
})

watch(() => props.gazePoints, () => init(), { deep: true })
</script>
