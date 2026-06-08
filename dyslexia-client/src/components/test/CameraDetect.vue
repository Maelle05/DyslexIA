<template>
  <div>
    <h2>Vérification de la caméra</h2>
    <p v-if="!mediapipeReady" style="color:orange">⏳ Chargement du modèle IA…</p>
    <p v-if="error" style="color:red">{{ error }}</p>
    <p v-if="ready" style="color:green">Caméra détectée ✓</p>
    <button :disabled="!ready || !mediapipeReady" @click="emit('next')">
      {{ !mediapipeReady ? 'Chargement…' : 'Continuer' }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps<{
  video: HTMLVideoElement | null
  mediapipeReady: boolean
}>()
const emit = defineEmits<{ next: [] }>()
const ready = ref(false)
const error = ref('')

onMounted(async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    if (props.video) {
      props.video.srcObject = stream
      await props.video.play()
    }
    ready.value = true
  } catch {
    error.value = "Impossible d'accéder à la caméra."
  }
})
</script>
