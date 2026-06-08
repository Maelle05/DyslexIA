<template>
  <div>
    <h2>Analyse</h2>

    <!-- Envoi en cours -->
    <div v-if="status === 'sending'">
      <p>⏳ Envoi des données…</p>
    </div>

    <!-- Erreur -->
    <div v-else-if="status === 'error'">
      <p style="color:red">{{ errorMsg }}</p>
      <button @click="send">Réessayer</button>
    </div>

    <!-- Résultat -->
    <div v-else-if="status === 'done' && result !== null">
      <p>✅ Analyse terminée</p>
      <RouterLink to="/results">Voir le rapport</RouterLink>
      <pre>{{ JSON.stringify(result, null, 2) }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'
import type { SessionData } from '@/types'
const store = useSessionStore()

const status = ref<'sending' | 'done' | 'error'>('sending')
const result = ref<any>(null)
const errorMsg = ref('')

/** Construit le CSV identique à ce que génère le Streamlit :
 *  colonnes : time, angle1_l, angle2_l, angle1_r, angle2_r */
function buildCsv(session: SessionData): string {
  const header = 'time,angle1_l,angle2_l,angle1_r,angle2_r'
  const rows = session.gazePoints.map(p =>
    `${p.t},${p.yaw_l},${p.pitch_l},${p.yaw_r},${p.pitch_r}`
  )
  return [header, ...rows].join('\n')
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
    console.log(csv)

    const response = await fetch('http://localhost:8000/predict', {
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
