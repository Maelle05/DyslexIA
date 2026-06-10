<template>
  <div class="bg-white p-4 shadow">
    <h3 class="font-semibold mb-4">Suivi du regard</h3>

    <Line v-if="chartData.length" :data="data" :options="options"/>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSessionStore } from '@/stores/session'

import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend
} from 'chart.js'

import { Line } from 'vue-chartjs'

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

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend)

const store = useSessionStore()

const chartData = computed(() => store.data?.gazePoints ?? [])

const data = computed(() => ({
  labels: chartData.value.map(p => p.t),
  datasets: [
    {
      label: 'écart gauche',
      data: chartData.value.map(p => p.yaw_l),
      borderColor: '#9966FF',
      backgroundColor: '#9966FF',
      tension: 0.3,
      pointRadius: 1.5,
    },
    {
      label: 'écart droite',
      data: chartData.value.map(p => p.yaw_r),
      borderColor: '#4BC0C0',
      backgroundColor: '#4BC0C0',
      tension: 0.3,
      pointRadius: 1.5,
    }
  ]
}))
</script>
