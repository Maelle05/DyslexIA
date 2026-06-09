<template>
  <div class="max-w-[700px] mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-10">Avertissement</h2>


      <p class="mb-6">
        Après un décompte de 3 secondes, un texte va s'afficher. <br>
        Lisez-le attentivement, une question de compréhension va vous être posée par la suite.
      </p>

      <button
        class="mt-6 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50"
        @click="emit('next')"
      >
        J'ai compris
      </button>
  </div>
</template>

<script setup lang="ts">
const emit = defineEmits<{ next: [] }>()
</script>
