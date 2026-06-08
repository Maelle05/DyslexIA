<template>
  <div>
    <h2>Préparez-vous</h2>
    <p style="font-size:4rem">{{ count }}</p>
    <p v-if="count === 0">C'est parti !</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const emit = defineEmits<{ done: [] }>()
const count = ref(3)

onMounted(() => {
  const interval = setInterval(() => {
    count.value--
    if (count.value <= 0) {
      clearInterval(interval)
      setTimeout(() => emit('done'), 500)
    }
  }, 1000)
})
</script>
