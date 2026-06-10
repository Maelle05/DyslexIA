<template>
  <div class="max-w-md mx-auto text-center px-4 py-8">
    <h2 class="text-3xl font-semibold mb-4 font-display" >Vérification de la caméra</h2>
      <p v-if="!mediapipeReady" class="text-orange-500" >⏳ Chargement du modèle IA…</p>
      <p v-if="error" class="text-red-500 mt-2">{{ error }}</p>
      <p v-if="ready" class="text-[#6E9960] bg-white inline-block px-4 py-2 mt-2">Caméra détectée ✓</p>
      <p v-if="ready" class="mt-7 mb-2">Après cette étape, veillez à bouger la tête le moins possible.</p>
      <div class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300 inline-block">
        <button v-if="ready" :disabled="!ready || !mediapipeReady" @click="emit('next')"
          class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                 before:absolute before:bottom-[0px] before:left-0
                 before:h-[100%] before:w-full
                 before:bg-[#299F9C] before:[-z-index:1]
                 before:scale-x-0 before:origin-left
                 before:transition-transform before:duration-300
                 hover:before:scale-x-100"
        >
          <span class="relative z-10 cursor-pointer">{{ !mediapipeReady ? 'Chargement…' : 'Continuer' }}</span>
        </button>
      </div>
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
