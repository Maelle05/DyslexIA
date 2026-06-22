<template>
  <div class="bg-white p-4 shadow">
    <h3 class="font-semibold mb-4">Relation X / Y (regard)</h3>
    <p class="text-center text-sm text-gray-400 mb-4">Trajectoire de la position du regard sur l'écran</p>
    <Scatter v-if="dataPoints.length" :data="chartData" :options="options"/>
    <p v-else class="text-gray-500">Aucune donnée disponible</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSessionStore } from '@/stores/session'

import {
  Chart as ChartJS,
  PointElement,
  LinearScale,
  Tooltip,
  Legend
} from 'chart.js'

import { Scatter } from 'vue-chartjs'

const options = {
  responsive: true,
  events: [],
  plugins: {
    tooltip: {
      enabled: false,
    },
    legend: {
      display: false,
    },
  },
}

ChartJS.register(PointElement, LinearScale, Tooltip, Legend)

const store = useSessionStore()

const dataPoints = computed(() => store.data?.gazePoints ?? [])

// Scatter format: { x: number, y: number }
const chartData = computed(() => ({
  datasets: [
    {
      label: 'Œil gauche',
      data: dataPoints.value.map(p => ({
        x: p.yaw_l,
        y: p.pitch_l
      })),
      borderColor: '#6E9960',
      backgroundColor: '#6E9960',
      showLine: true,
      pointRadius: 1.5,
    },
    {
      label: 'Œil droit',
      data: dataPoints.value.map(p => ({
        x: p.yaw_r,
        y: p.pitch_r
      })),
      borderColor: '#E86A64',
      backgroundColor: '#E86A64',
      showLine: true,
      pointRadius: 1.5,
    }
  ]
}))
</script>
