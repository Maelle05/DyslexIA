<template>
  <div class="max-w-md mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-10">Préparez-vous</h2>
    <p v-if="count >= 1" class="text-7xl font-bold font-display tabular-nums">{{ count}}</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const emit = defineEmits<{ done: [] }>()
const count = ref(3)

onMounted(() => {
  const interval = setInterval(() => {
    count.value--
    if (count.value < 0) {
      clearInterval(interval)
      setTimeout(() => emit('done'), 500)
    }
  }, 1000)
})
</script>
