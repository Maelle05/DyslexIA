<template>
  <div>
    <p>Étape {{ stepIndex + 1 }} / {{ steps.length }} — {{ step }}</p>

    <!-- Preview caméra toujours visible -->
    <video ref="videoRef" autoplay playsinline muted style="width:160px; height:120px;" />

    <CameraDetect   v-if="step === 'camera'"  :video="videoRef" :mediapipe-ready="faceLandmarkerReady"  @next="onCameraReady" />
    <Calibration    v-if="step === 'calibration'" @next="next" />
    <Countdown      v-if="step === 'countdown'"   @done="onCountdownDone" />
    <ReadingSession v-if="step === 'reading'"
      :face-landmarker="faceLandmarker"
      :video="videoRef"
      :reading-text="readingText"
      @stop="onStop"
    />
    <PredictionView v-if="step === 'prediction'" :session="sessionData" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, inject, onMounted, onUnmounted, markRaw} from 'vue'
import type { Ref } from 'vue'
import type { TestStep, SessionData } from '@/types'
import CameraDetect   from '@/components/test/CameraDetect.vue'
// import Calibration    from '@/components/test/Calibration.vue'
import Countdown      from '@/components/test/Countdown.vue'
import ReadingSession from '@/components/test/ReadingSession.vue'
import PredictionView from '@/components/test/PredictionView.vue'

const sessionData = inject<Ref<SessionData | null>>('sessionData', ref(null))

// ── Steps ──────────────────────────────────────────────────────────────────
const steps: TestStep[] = ['camera', 'countdown', 'reading', 'prediction']
// const steps: TestStep[] = ['camera', 'calibration', 'countdown', 'reading', 'prediction']
const stepIndex = ref(0)
const step = computed(() => steps[stepIndex.value])
function next() { if (stepIndex.value < steps.length - 1) stepIndex.value++ }

// ── Texte de lecture ───────────────────────────────────────────────────────
const readingText = `Le soleil se levait lentement sur la ville endormie. Marie ouvrit les yeux et regarda le plafond blanc de sa chambre. Elle aimait ces moments tranquilles du matin, avant que le monde ne s'éveille vraiment. Sur sa table de nuit, un livre attendait patiemment depuis trois jours.`

// ── Caméra + MediaPipe ────────────────────────────────────────────────────
const videoRef = ref<HTMLVideoElement | null>(null)
const faceLandmarker = ref<any>(null)
let stream: MediaStream | null = null

async function onCameraReady() {
  next()
}

const faceLandmarkerReady = ref(false)

onMounted(async () => {
  try {
    const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision')
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    )
    const fl = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    })
    faceLandmarker.value = markRaw(fl) // ← markRaw empêche Vue de proxifier l'objet
    faceLandmarkerReady.value = true
  } catch (e) {
    console.error('MediaPipe init error:', e)
  }
})

onUnmounted(() => {
  stream?.getTracks().forEach(t => t.stop())
  faceLandmarker.value?.close?.()
})

// ── Décompte terminé ───────────────────────────────────────────────────────
function onCountdownDone() {
  next() // passe à 'reading'
}

// ── Arrêt session ──────────────────────────────────────────────────────────
function onStop(data: SessionData) {
  sessionData.value = data
  next()
}
</script>
