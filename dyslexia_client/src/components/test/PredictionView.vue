<template>
  <div class="max-w-6xl mx-auto text-center px-4 pt-2 pb-8">
    <h2 class="text-3xl font-semibold mb-10 font-display">Analyse</h2>

    <!-- Envoi en cours -->
    <div v-if="status === 'sending'" class="text-gray-600">
      <p class="text-orange-500 bg-white inline-block px-4 py-2 mt-2">⏳ Envoi des données…</p>
    </div>

    <!-- Erreur -->
    <div v-else-if="status === 'error'" class="text-red-600">
      <p class="text-red-500 bg-white inline-block px-4 py-2 mt-2">{{ errorMsg }}</p>
      <button @click="send" class="bg-white px-5 py-2 transition hover:bg-[#E86A64] active:scale-95">Réessayer</button>
    </div>

    <!-- Résultat -->
    <div v-else-if="status === 'done' && result !== null" class="space-y-6">
      <p class="text-green-500 bg-white inline-block px-4 py-2 mt-2 shadow" >✅ Analyse terminée</p>

      <div class="text-3xl bg-white p-4 overflow-auto font-bold shadow">
        <p>{{ result.Prediction }} à {{ Math.round(result.Probability * 100) }}%</p>
      </div>

      <!-- <RouterLink to="/results"
        class="inline-block border rounded-full px-6 py-2 hover:bg-gray-50 transition"
        >Voir le rapport</RouterLink> -->
      <GazeHeatmap
        v-if="store.data"
        :text="store.data.readingText ?? ''"
        :gaze-points="store.data.gazePoints"
        class="mt-4"
      />
      <GazeXYChart />
      <GazeChart />

      <br>

      <div class="rotate-[2deg] hover:scale-105 hover:-rotate-[3deg] transition-transform duration-300 inline-block">
      <button @click="downloadCSV" class="mt-8 cursor-pointer text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
                    before:absolute before:bottom-[0px] before:left-0
                    before:h-[100%] before:w-full
                    before:bg-[#299F9C] before:[-z-index:1]
                    before:scale-x-0 before:origin-left
                    before:transition-transform before:duration-300
                    hover:before:scale-x-100"
                    > <span class="relative z-10 cursor-pointer"> Télécharger les données de suivi oculaire </span></button>
                    </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'
import type { SessionData } from '@/types/index'
import GazeChart   from '@/components/chart/GazeChart.vue'
import GazeXYChart   from '@/components/chart/GazeXYChart.vue'
import GazeHeatmap from '@/components/chart/GazeHeatmap.vue'

const store = useSessionStore()

const status = ref<'sending' | 'done' | 'error'>('sending')
const result = ref<any>(null)
const errorMsg = ref('')

/** Construit le CSV identique à ce que génère le Streamlit :
 *  colonnes : time, angle1_l, angle2_l, angle1_r, angle2_r */
function buildCsv(session: SessionData): string {
  const header = 'time,x_left,y_left,x_right,y_right'
  const rows = session.gazePoints.map(p =>
    `${p.t},${p.yaw_l},${p.pitch_l},${p.yaw_r},${p.pitch_r}`
  )
  return [header, ...rows].join('\n')
}

function downloadCSV() {
  if (!store.data) {
    errorMsg.value = 'Aucune session disponible.'
    status.value = 'error'
    return
  }

  const csv = buildCsv(store.data)

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)

  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', 'gaze_data.csv')

  document.body.appendChild(link)
  link.click()

  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

async function send() {
  if (!store.data) {
    errorMsg.value = 'Aucune session disponible.'
    status.value = 'error'
    return
  }

  status.value = 'sending'

  try {
    const csv = buildCsv(store.data)
    const blob = new Blob([csv], { type: 'text/csv' })
    const formData = new FormData()
    formData.append('csv_file', blob, 'gaze_log.csv')
    // console.log(csv)

    const response = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Erreur ${response.status}: ${await response.text()}`)
    }

    result.value = await response.json()
    console.log(result.value)
    status.value = 'done'
  } catch (e: any) {
    errorMsg.value = e.message ?? 'Erreur inconnue'
    status.value = 'error'
  }
}

onMounted(() => send())
</script>
