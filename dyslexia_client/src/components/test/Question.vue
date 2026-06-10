<template>
  <div class="max-w-[700px] mx-auto text-center px-4 pb-8 pt-2">
    <h2 class="text-3xl font-semibold mb-4 font-display">Question</h2>

    <div class="h-[calc(100vh-200px)] flex flex-col align-center justify-center">
      <p class="mb-8 text-lg">{{ questionText }}</p>

      <div class="flex flex-col gap-3 mb-8">
        <button
          v-for="(answer, index) in [a1Text, a2Text, a3Text]"
          :key="index"
          :class="[
            'px-5 py-3 transition cursor-pointer text-left shadow active:scale-95',
            selected === index
              ? 'bg-[#6E9960E6]'
              : 'bg-white hover:bg-[#299F9CE6]'
          ]"
          @click="selectedAnswer(index)"
        >
          <span class="font-semibold mr-2">{{ index + 1 }}.</span>{{ answer }}
        </button>
      </div>

      <!-- Résultat après validation -->
      <div v-if="validated" class="mb-2">
        <p v-if="selected === aText" class="text-[#6E9960] bg-white inline-block px-4 py-2 mt-2">
          ✓ Bonne réponse !
        </p>
        <p v-else class="text-[#E86A64] bg-white inline-block px-4 py-2 mt-2">
          ✗ Mauvaise réponse... la bonne réponse était : {{ [a1Text, a2Text, a3Text][aText] }}.
        </p>
      </div>

      <div v-if="!validated" class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300 inline-block">
        <button
          class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                      before:absolute before:bottom-[0px] before:left-0
                      before:h-[100%] before:w-full
                      before:bg-[#299F9C] before:[-z-index:1]
                      before:scale-x-0 before:origin-left
                      before:transition-transform before:duration-300
                      hover:before:scale-x-100"
          :disabled="selected === null"
          @click="validate"
        >
          <span class="relative z-10 cursor-pointer">Valider</span>
        </button>
      </div>

      <div v-if="validated && selected === aText" class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300 inline-block">
        <button
          class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                    before:absolute before:bottom-[0px] before:left-0
                    before:h-[100%] before:w-full
                    before:bg-[#299F9C] before:[-z-index:1]
                    before:scale-x-0 before:origin-left
                    before:transition-transform before:duration-300
                    hover:before:scale-x-100"
          :disabled="selected === null"
          @click="validate"
        >
          <span class="relative z-10 cursor-pointer">Continuer</span>
        </button>
      </div>

      <div v-if="validated && selected != aText" class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300 inline-block">
        <button
          class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                    before:absolute before:bottom-[0px] before:left-0
                    before:h-[100%] before:w-full
                    before:bg-[#299F9C] before:[-z-index:1]
                    before:scale-x-0 before:origin-left
                    before:transition-transform before:duration-300
                    hover:before:scale-x-100"
          :disabled="selected === null"
          @click="returnRead"
        >
          <span class="relative z-10 cursor-pointer">Recomencer</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  questionText: string
  a1Text: string
  a2Text: string
  a3Text: string
  aText: number  // id de la bonne réponse (1, 2 ou 3)
}>()

const emit = defineEmits<{ next: [correct: boolean], returnRead: [correct: boolean] }>()

const selected = ref<number | null>(null)
const validated = ref(false)

function validate() {
  if (validated.value) {
    emit('next', selected.value === props.aText)
    return
  }
  validated.value = true
}

function selectedAnswer(index: number) {
  if(validated.value == false){
    selected.value = index
  }
}

function returnRead() {
  emit('returnRead', true)
}
</script>
