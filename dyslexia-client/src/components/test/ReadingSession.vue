<template>
  <div class="max-w-6xl mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-10" >Lecture en cours</h2>
    <p class="text-lg text-left whitespace-pre-line leading-loose max-w-3xl mx-auto mb-8">{{ formattedText(readingText) }}</p>
    <button
    class="mt-6 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50"
    @click="stop">⏹ Arrêter et analyser</button>
    <p class="mt-6 text-sm text-gray-600">⏱ {{ elapsed }}s — {{ gazePoints.length }} points enregistrés</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { GazePoint, SessionData } from '@/types'

const props = defineProps<{
  video: HTMLVideoElement | null
  faceLandmarker: any
  readingText: string
}>()

const emit = defineEmits<{ stop: [data: SessionData] }>()

const elapsed = ref(0)
const gazePoints = ref<GazePoint[]>([])

let animId: number
let timerInterval: ReturnType<typeof setInterval>
let startTime = 0

function formattedText(text) {
  return text.replaceAll('.', '.\n')
}

// ── Maths utilitaires ─────────────────────────────────────────────────────

function norm3(v: number[]) {
  const l = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
  return l > 0 ? v.map(x => x / l) : v
}

function dot3(a: number[], b: number[]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function clamp(x: number, min: number, max: number) {
  return Math.max(min, Math.min(max, x))
}

/** Convertit un vecteur de regard 3D en (yaw_deg, pitch_deg).
 *  Positif yaw = droite, positif pitch = bas — identique à la version Python. */
function gazeToAngles(d: number[]): [number, number] {
  d = norm3(d)

  // Yaw : projection sur le plan XZ
  const xz = norm3([d[0], 0, d[2]])
  let yaw = Math.acos(clamp(dot3([0, 0, -1], xz), -1, 1))
  if (d[0] < 0) yaw = -yaw

  // Pitch : projection sur le plan YZ
  const yz = norm3([0, d[1], d[2]])
  let pitch = Math.acos(clamp(dot3([0, 0, -1], yz), -1, 1))
  if (d[1] > 0) pitch = -pitch

  return [-yaw * (180 / Math.PI), pitch * (180 / Math.PI)]
}

/** Centroïde 3D d'un ensemble de landmarks. */
function centroid(pts: number[][]): number[] {
  const n = pts.length
  return [
    pts.reduce((s, p) => s + p[0], 0) / n,
    pts.reduce((s, p) => s + p[1], 0) / n,
    pts.reduce((s, p) => s + p[2], 0) / n,
  ]
}

/** Scale proxy : distance moyenne entre points (comme compute_scale Python). */
function computeScale(pts: number[][]): number {
  let total = 0, count = 0
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const d = pts[i].map((v, k) => v - pts[j][k])
      total += Math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
      count++
    }
  }
  return count > 0 ? total / count : 1
}

/** Indices des landmarks du nez utilisés pour estimer la pose de la tête
 *  (correspondant à NOSE_INDICES dans utils.py). */
const NOSE_INDICES = [1, 2, 4, 5, 6, 19, 20, 94, 125, 141, 235, 281, 354, 370, 460]

// ── État de calibration automatique (init au 1er frame) ───────────────────
let initialised = false
let loff: number[] | null = null
let roff: number[] | null = null
let lcal = 1
let rcal = 1
const BASE_R = 20

// ── Boucle de détection ───────────────────────────────────────────────────

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

        // Points du nez pour estimer la tête
        const nosePts = NOSE_INDICES.map(i => [
          lms[i].x * w,
          lms[i].y * h,
          lms[i].z * w,
        ])

        const hc = centroid(nosePts)
        const curScale = computeScale(nosePts)

        // Iris : 468 = gauche, 473 = droite (mediapipe tasks-vision)
        const il = [lms[468].x * w, lms[468].y * h, lms[468].z * w]
        const ir = [lms[473].x * w, lms[473].y * h, lms[473].z * w]

        // Auto-init au premier frame (même logique que Python)
        if (!initialised) {
          const forward = [0, 0, 1]
          loff = il.map((v, k) => v - hc[k] + BASE_R * forward[k])
          roff = ir.map((v, k) => v - hc[k] + BASE_R * forward[k])
          lcal = curScale
          rcal = curScale
          initialised = true
        }

        // Sources estimées des yeux selon la pose courante
        const sl = hc.map((v, k) => v + (loff![k] * (curScale / lcal)))
        const sr = hc.map((v, k) => v + (roff![k] * (curScale / rcal)))

        // Vecteurs de regard
        const dl = norm3(il.map((v, k) => v - sl[k]))
        const dr = norm3(ir.map((v, k) => v - sr[k]))

        const [yaw_l, pitch_l] = gazeToAngles(dl)
        const [yaw_r, pitch_r] = gazeToAngles(dr)

        gazePoints.value.push({
          t: Date.now(),
          // Position normalisée moyenne des deux iris (0-1)
          x: (lms[468].x + lms[473].x) / 2,
          y: (lms[468].y + lms[473].y) / 2,
          // Position normalisée individuelle
          x_l: lms[468].x,
          y_l: lms[468].y,
          x_r: lms[473].x,
          y_r: lms[473].y,
          // Angles calculés
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

// ── Calcul métriques à l'arrêt ────────────────────────────────────────────

function stop() {
  clearInterval(timerInterval)
  cancelAnimationFrame(animId)

  const points = gazePoints.value

  console.log(points)

  emit('stop', { gazePoints: points })
}
</script>
