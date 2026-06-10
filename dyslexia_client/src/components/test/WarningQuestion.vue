<template>
  <div class="max-w-[700px] mx-auto text-center px-4 py-8">
    <h2 class="text-3xl font-semibold mb-10 font-display">Avertissement</h2>


      <p class="mb-6">
        Après un décompte de 3 secondes, un texte va s'afficher. <br>
        Lisez-le attentivement, une question de compréhension va vous être posée par la suite.
      </p>

      <div class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300 inline-block">
        <button
          class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                     before:absolute before:bottom-[0px] before:left-0
                     before:h-[100%] before:w-full
                     before:bg-[#299F9C] before:[-z-index:1]
                     before:scale-x-0 before:origin-left
                     before:transition-transform before:duration-300
                     hover:before:scale-x-100"
          @click="emit('next')"
        >
          <span class="relative z-10 cursor-pointer">J'ai compris</span>
        </button>
      </div>
  </div>
</template>

<script setup lang="ts">
const emit = defineEmits<{ next: [] }>()
</script>
