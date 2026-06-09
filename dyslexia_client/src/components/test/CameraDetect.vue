<template>
  <div class="max-w-md mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-4" >Vérification de la caméra</h2>
    <p v-if="!mediapipeReady" class="text-orange-500" >⏳ Chargement du modèle IA…</p>
    <p v-if="error" class="text-red-500 mt-2">{{ error }}</p>
    <p v-if="ready" class="text-green-600 mt-2">Caméra détectée ✓</p>
    <p v-if="ready" class="mt-2">Après cette étape, veillez à bouger la tête le moins possible.</p>
    <button v-if="ready" :disabled="!ready || !mediapipeReady" @click="emit('next')"
      class="mt-6 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50"
    >
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
