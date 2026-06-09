<template>
  <div class="max-w-6xl mx-auto text-center px-4 py-8">
    <h2 class="text-2xl font-semibold mb-8">Analyse</h2>

    <!-- Envoi en cours -->
    <div v-if="status === 'sending'" class="text-gray-600">
      <p>⏳ Envoi des données…</p>
    </div>

    <!-- Erreur -->
    <div v-else-if="status === 'error'" class="text-red-600">
      <p class="mb-4">{{ errorMsg }}</p>
      <button @click="send" class="rounded-lg border px-5 py-2 transition hover:bg-red-50 active:scale-95">Réessayer</button>
    </div>

    <!-- Résultat -->
    <div v-else-if="status === 'done' && result !== null" class="space-y-6">
      <p class="text-green-600 font-medium" >✅ Analyse terminée</p>

      <div class="mt-10 text-xl bg-gray-50 p-4 rounded-lg overflow-auto font-semibold">
        <p>{{ result.Prediction === 0 ? 'Non dyslexique' : 'Dyslexique' }} à {{ Math.round(result.Probability * 100) }}%</p>
      </div>

      <!-- <RouterLink to="/results"
        class="inline-block border rounded-full px-6 py-2 hover:bg-gray-50 transition"
        >Voir le rapport</RouterLink> -->
      <GazeXYChart />
      <GazeChart />

      <br>
      <button @click="downloadCSV" class="rounded-lg border px-5 py-2 transition hover:bg-red-50 active:scale-95 cursor-pointer">Télécharger les données de suivi oculaire</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'
import type { SessionData } from '@/types/index'
import GazeChart   from '@/components/chart/GazeChart.vue'
import GazeXYChart   from '@/components/chart/GazeXYChart.vue'

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
    status.value = 'done'
  } catch (e: any) {
    errorMsg.value = e.message ?? 'Erreur inconnue'
    status.value = 'error'
  }
}

onMounted(() => send())
</script>
