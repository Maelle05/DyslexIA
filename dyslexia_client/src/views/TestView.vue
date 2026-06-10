<template>
  <p class="absolute top-12 left-[180px] text-center text-black underline underline-offset-4">Step {{ stepIndex + 1 }} / {{ steps.length }} :</p>
  <div class="max-w-4xl mx-auto px-4 py-8 mt-15">

    <!-- Preview caméra toujours visible -->
    <div
    class="fixed z-50 transition-all duration-1000 bg-[#E9E4DB]"
    :class="step === 'camera'
        ? 'top-[150px] right-1/2 translate-x-1/2'
        : step === 'calibration' ? 'top-[calc(100vh-42vh)] right-4' : 'top-[calc(100vh-18vh)] right-4'"
    >
      <video ref="videoRef" autoplay playsinline muted
        class="rounded-lg border shadow-lg transition-all duration-1000 shadow-xl ring-2 ring-black/10"
        :class="(step === 'camera' || step === 'calibration')
          ? 'h-[40vh] w-auto'
          : 'h-[15vh] w-auto'"
        />
    </div>

    <CameraDetect   v-if="step === 'camera'"  :video="videoRef" :mediapipe-ready="faceLandmarkerReady"  @next="onCameraReady" />
    <Calibration
      v-if="step === 'calibration'"
      :video="videoRef"
      :face-landmarker="faceLandmarker"
      @next="onCalibrationDone"
    />
    <WarningQuestion v-if="step === 'warning-question'" @next="next"/>
    <Countdown      v-if="step === 'countdown'"   @done="onCountdownDone" />
    <ReadingSession
      v-if="step === 'reading'"
      :face-landmarker="faceLandmarker"
      :video="videoRef"
      :reading-text="readingText"
      :calib="calibData"
      @stop="onStop"
    />
    <Question v-if="step === 'question'"
      :question-text="questionText"
      :a1-text="a1Text"
      :a2-text="a2Text"
      :a3-text="a3Text"
      :a-text="parseInt(aText)" @next="next" @returnRead="returnRead"
    />
    <PredictionView v-if="step === 'prediction'" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, markRaw } from 'vue'
import { useSessionStore } from '@/stores/session'
import type { TestStep, SessionData, CalibrationData} from '@/types/index'
const calibData = ref<CalibrationData | null>(null)
import CameraDetect   from '@/components/test/CameraDetect.vue'
import Calibration    from '@/components/test/Calibration.vue'
import WarningQuestion    from '@/components/test/WarningQuestion.vue'
import Question    from '@/components/test/Question.vue'
import Countdown      from '@/components/test/Countdown.vue'
import ReadingSession from '@/components/test/ReadingSession.vue'
import PredictionView from '@/components/test/PredictionView.vue'

const store = useSessionStore()

// ── Steps ──────────────────────────────────────────────────────────────────
const steps: TestStep[] = ['camera', 'calibration', 'warning-question', 'countdown', 'reading', 'question', 'prediction']
const stepIndex = ref(0)
const step = computed(() => steps[stepIndex.value])
function next() { if (stepIndex.value < steps.length - 1) stepIndex.value++ }
function returnRead() {
  store.clear()
  stepIndex.value = steps.indexOf('countdown')
}

// ── Texte de lecture ───────────────────────────────────────────────────────
const readingText = ref('Chargement du texte…')
const questionText = ref('Chargement de la question…')
const a1Text = ref('Chargement de la question…')
const a2Text = ref('Chargement de la question…')
const a3Text = ref('Chargement de la question…')
const aText = ref('Chargement de la question…')

// ── Caméra + MediaPipe ────────────────────────────────────────────────────
const videoRef = ref<HTMLVideoElement | null>(null)
const faceLandmarker = ref<any>(null)
let stream: MediaStream | null = null

async function onCameraReady() {
  stream = videoRef.value?.srcObject as MediaStream ?? null
  next()
}

const faceLandmarkerReady = ref(false)

function onCalibrationDone(data: CalibrationData) {
  calibData.value = data
  next()
}

onMounted(async () => {
  // Fetch du texte
  try {
    console.log(`${import.meta.env.VITE_API_URL}/text-question`)
    const response = await fetch(`${import.meta.env.VITE_API_URL}/text-question`, { method: 'GET' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    readingText.value = data.text ?? 'Texte non disponible'
    questionText.value = data.query.q ?? '? non disponible'
    a1Text.value = data.query.a1 ?? '? non disponible'
    a2Text.value = data.query.a2 ?? '? non disponible'
    a3Text.value = data.query.a3 ?? '? non disponible'
    aText.value = data.query.valid ?? 0
  } catch (e) {
    console.error('Texte error:', e)
    readingText.value = 'Erreur lors du chargement du texte.'
  }

  // Init MediaPipe
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
  stream = null

  // Double sécurité via l'élément vidéo
  if (videoRef.value) {
    const s = videoRef.value.srcObject as MediaStream | null
    s?.getTracks().forEach(t => t.stop())
    videoRef.value.srcObject = null
  }

  faceLandmarker.value?.close?.()
})

// ── Décompte terminé ───────────────────────────────────────────────────────
function onCountdownDone() {
  next()
}

// ── Arrêt session ──────────────────────────────────────────────────────────
function onStop(data: SessionData) {
  store.save(data)
  next()
}
</script>
