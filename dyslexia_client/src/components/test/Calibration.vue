<template>
  <div class="max-w-md mx-auto text-center px-4 py-8">
    <h2 class="text-3xl font-semibold mb-5 font-display">Calibration</h2>

    <p v-if="!faceLocked" class="text-[#E86A64] bg-white inline-block px-4 py-2 mt-2">
      ⚠ Aucun visage détecté
    </p>
    <p v-else-if="calibStep === 'lock'" class="text-[#6E9960] bg-white inline-block px-4 py-2 mt-2">
      ✓ Visage détecté
    </p>

    <!-- Étape 1 : lock des sphères oculaires -->
    <div v-if="calibStep === 'lock' && faceLocked" class="mt-10">
      <p class="mb-6">Regardez le point rouge et cliquez dessus</p>
      <button
        class="border w-10 h-10 bg-red-300 border-red-600 rounded-full hover:bg-red-600 cursor-pointer"
        @click="lockSpheres"
      />
    </div>

    <!-- Étape done -->
    <div v-else-if="calibStep === 'done'">
      <p class="text-[#6E9960] bg-white inline-block px-4 py-2 mt-2">✓ Calibration complète</p>
      <br />
      <div class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300 inline-block">
        <button
          class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                   before:absolute before:bottom-[0px] before:left-0
                   before:h-[100%] before:w-full
                   before:bg-[#299F9C] before:[-z-index:1]
                   before:scale-x-0 before:origin-left
                   before:transition-transform before:duration-300
                   hover:before:scale-x-100"
          @click="emit('next', calibData!)"
        >
          <span class="relative z-10 cursor-pointer">Continuer</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { CalibrationData } from '@/types/index'
import {
  BASE_R,
  centroid,
  computeScale,
  computeHeadRotation,
  matTransposeVec,
  extractNosePts,
  extractIris,
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
