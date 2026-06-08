<template>
  <div class="max-w-6xl mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-10">Lecture en cours</h2>
    <p class="text-lg text-left whitespace-pre-line leading-loose max-w-3xl mx-auto mb-8">
      {{ formattedText(readingText) }}
    </p>
    <button
      class="mt-6 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50"
      @click="stop"
    >
      ⏹ Arrêter et analyser
    </button>
    <p class="mt-6 text-sm text-gray-600">⏱ {{ elapsed }}s — {{ gazePoints.length }} points enregistrés</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { GazePoint, SessionData, CalibrationData } from '@/types'
import {
  centroid,
  computeScale,
  extractNosePts,
  extractIris,
  computeEyeSpheres,
  computeGazeAngles,
} from '@/lib/gazeUtils'

const props = defineProps<{
  video: HTMLVideoElement | null
  faceLandmarker: any
  readingText: string
  calib: CalibrationData | null
}>()

const emit = defineEmits<{ stop: [data: SessionData] }>()

const elapsed = ref(0)
const gazePoints = ref<GazePoint[]>([])

let animId: number
let timerInterval: ReturnType<typeof setInterval>
let startTime = 0

function formattedText(text: string): string {
  return text.replaceAll('.', '.\n')
}

onMounted(() => {
  startTime = Date.now()

  timerInterval = setInterval(() => {
    elapsed.value = Math.floor((Date.now() - startTime) / 1000)
  }, 1000)

  const detect = () => {
    const fl = props.faceLandmarker
    const vid = props.video

    if (!fl || !vid || vid.readyState < 2) {
      animId = requestAnimationFrame(detect)
      return
    }

    try {
      const results = fl.detectForVideo(vid, performance.now())
      const lms = results.faceLandmarks?.[0]

      if (lms) {
        const w = vid.videoWidth
        const h = vid.videoHeight

        const nosePts = extractNosePts(lms, w, h)
        const hc = centroid(nosePts)
        const curScale = computeScale(nosePts)
        const [il, ir] = extractIris(lms, w, h)
        const [sl, sr] = computeEyeSpheres(hc, il, ir, curScale, props.calib)
        const { yaw_l, pitch_l, yaw_r, pitch_r } = computeGazeAngles(il, ir, sl, sr, props.calib)

        gazePoints.value.push({
          t: Date.now() - startTime,
          x:   (lms[468].x + lms[473].x) / 2,
          y:   (lms[468].y + lms[473].y) / 2,
          x_l: lms[468].x,
          y_l: lms[468].y,
          x_r: lms[473].x,
          y_r: lms[473].y,
          yaw_l,
          pitch_l,
          yaw_r,
          pitch_r,
        })
      }
    } catch (e) {
      console.error('detectForVideo error:', e)
    }

    animId = requestAnimationFrame(detect)
  }

  detect()
})

onUnmounted(() => {
  clearInterval(timerInterval)
  cancelAnimationFrame(animId)
})

function stop() {
  clearInterval(timerInterval)
  cancelAnimationFrame(animId)
  console.log(gazePoints.value)
  emit('stop', { gazePoints: gazePoints.value })
}
</script>
