<template>
  <div class="max-w-md mx-auto text-center px-4 pt-2 pb-8">
    <h2 class="text-3xl font-semibold mb-4 font-display">Entrez votre pseudo</h2>

    <div class="mt-[25vh]">
      <div class="">
        <input
          v-model="pseudo"
          type="text"
          placeholder="Votre pseudo…"
          class="border-2 border-[#299F9C] px-4 py-2 text-lg w-full max-w-xs focus:outline-none focus:border-[#E86A64] bg-[#E9E4DB] font-bold placeholder-gray-500"
        />
      </div>
      <div class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300 inline-block">
        <button
          @click="handleNext"
          class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                 before:absolute before:bottom-[0px] before:left-0
                 before:h-[100%] before:w-full
                 before:bg-[#299F9C] before:[-z-index:1]
                 before:scale-x-0 before:origin-left
                 before:transition-transform before:duration-300
                 hover:before:scale-x-100
                 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <span class="relative z-10 cursor-pointer">{{ 'Continuer' }}</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits(['next'])

const pseudo = ref('')

function handleNext() {
  if (!pseudo.value.trim()) return
  emit('next', pseudo.value.trim())
}
</script>
