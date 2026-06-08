<template>
  <div class="max-w-md mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-10">Calibration</h2>

    <p v-if="!faceLocked" class="text-amber-600 text-sm bg-amber-50 px-4 py-2 rounded-full inline-block mb-6">
      ⚠ Aucun visage détecté
    </p>
    <p v-else-if="calibStep === 'lock'" class="text-emerald-600 text-sm bg-emerald-50 px-4 py-2 rounded-full inline-block mb-6">
      ✓ Visage détecté
    </p>

    <!-- Étape 1 : lock des sphères oculaires -->
    <div v-if="calibStep === 'lock' && faceLocked">
      <p class="mb-6">Regardez au centre de l'écran, cliquez sur le <strong>point rouge</strong></p>
      <button
        class="border w-10 h-10 bg-red-100 border-red-600 rounded-full hover:bg-red-600 cursor-pointer"
        @click="lockSpheres"
      />
    </div>

    <!-- Étape done -->
    <div v-else-if="calibStep === 'done'">
      <p class="text-emerald-600 text-sm bg-emerald-50 px-4 py-2 rounded-full inline-block mb-6">✓ Calibration complète</p>
      <br />
      <button
        class="mt-6 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50"
        @click="emit('next', calibData!)"
      >
        Continuer
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { CalibrationData } from '@/types'
import {
  BASE_R,
  centroid,
  computeScale,
  computeHeadRotation,
  matTransposeVec,
  extractNosePts,
  extractIris,
  gazeToAngles,
  computeEyeSpheres,
  computeGazeAngles,
} from '@/lib/gazeUtils'

const props = defineProps<{
  video: HTMLVideoElement | null
  faceLandmarker: any
}>()

const emit = defineEmits<{ next: [data: CalibrationData] }>()

type CalibStep = 'lock' | 'done'
const calibStep = ref<CalibStep>('lock')
const faceLocked = ref(false)
const calibData = ref<CalibrationData | null>(null)

let lastLandmarks: any[] | null = null
let animId: number

// ── Boucle de détection ───────────────────────────────────────────────────

onMounted(() => {
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
      faceLocked.value = !!lms
      lastLandmarks = lms ?? null
    } catch { /* ignore */ }
    animId = requestAnimationFrame(detect)
  }
  detect()
})

onUnmounted(() => cancelAnimationFrame(animId))

// ── Lock sphères + calibration centre en une seule action ─────────────────

function lockSpheres() {
  if (!lastLandmarks) return
  const lms = lastLandmarks
  const vid = props.video!
  const w = vid.videoWidth
  const h = vid.videoHeight

  const nosePts = extractNosePts(lms, w, h)
  const hc = centroid(nosePts)
  const scale = computeScale(nosePts)
  const R = computeHeadRotation(nosePts)
  const [il, ir] = extractIris(lms, w, h)
  const camDirLocal = matTransposeVec(R, [0, 0, 1])

  const leftOffset  = matTransposeVec(R, il.map((v, k) => v - hc[k])).map((v, k) => v + BASE_R * camDirLocal[k])
  const rightOffset = matTransposeVec(R, ir.map((v, k) => v - hc[k])).map((v, k) => v + BASE_R * camDirLocal[k])

  calibData.value = {
    leftSphereOffset:  leftOffset,
    rightSphereOffset: rightOffset,
    leftCalibScale:    scale,
    rightCalibScale:   scale,
    yawOffset:   0,
    pitchOffset: 0,
  }

  // Calibration centre écran immédiatement après le lock
  const [sl, sr] = computeEyeSpheres(hc, il, ir, scale, calibData.value)
  const { yaw_l, pitch_l, yaw_r, pitch_r } = computeGazeAngles(il, ir, sl, sr, null)

  calibData.value.yawOffset   = -((yaw_l + yaw_r) / 2)
  calibData.value.pitchOffset = -((pitch_l + pitch_r) / 2)

  calibStep.value = 'done'
}
</script>
