<template>
  <div class="bg-white p-4 rounded-lg shadow">
    <h3 class="font-semibold mb-4">Relation X / Y (regard)</h3>

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
  interaction: {
    mode: null
  },
  plugins: {
    tooltip: {
      enabled: false
    },
    legend: {
      display: true
    }
  },
  hover: {
    mode: null
  }
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
      borderColor: '#9966FF',
      backgroundColor: '#9966FF',
      showLine: true,
      pointRadius: 1.5,
    },
    {
      label: 'Œil droit',
      data: dataPoints.value.map(p => ({
        x: p.yaw_r,
        y: p.pitch_r
      })),
      borderColor: '#4BC0C0',
      backgroundColor: '#4BC0C0',
      showLine: true,
      pointRadius: 1.5,
    }
  ]
}))
</script>
