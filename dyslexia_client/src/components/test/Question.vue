<template>
  <div class="max-w-[700px] mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-10">Question</h2>

    <p class="mb-8 text-lg">{{ questionText }}</p>

    <div class="flex flex-col gap-3 mb-8">
      <button
        v-for="(answer, index) in [a1Text, a2Text, a3Text]"
        :key="index"
        :class="[
          'px-5 py-3 bg-gray-50 rounded-full border transition cursor-pointer text-left',
          selected === index
            ? 'border-gray-400 text-gray-800'
            : 'hover:bg-gray-200 border-gray-200'
        ]"
        @click="selectedAnswer(index)"
      >
        <span class="font-semibold mr-2">{{ index + 1 }}.</span>{{ answer }}
      </button>
    </div>

    <!-- Résultat après validation -->
    <div v-if="validated" class="mb-6">
      <p v-if="selected === aText" class="text-emerald-600 bg-emerald-50 px-4 py-2 rounded-full inline-block">
        ✓ Bonne réponse !
      </p>
      <p v-else class="text-red-500 bg-red-50 px-4 py-2 rounded-full inline-block">
        ✗ Mauvaise réponse... la bonne réponse était : {{ [a1Text, a2Text, a3Text][aText] }}.
      </p>
    </div>

    <button v-if="!validated"
      class="mt-2 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
      :disabled="selected === null"
      @click="validate"
    >
      Valider
    </button>

    <button v-if="validated && selected === aText"
      class="mt-2 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
      :disabled="selected === null"
      @click="validate"
    >
      Continuer
    </button>
    <button v-if="validated && selected != aText"
      class="mt-2 rounded-full px-5 py-2 border transition cursor-pointer hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
      :disabled="selected === null"
      @click="returnRead"
    >
      Recomencer
    </button>
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
