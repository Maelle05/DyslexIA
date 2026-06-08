<template>
  <nav class="fixed top-0 w-full z-50 glass">
    <div class="max-w-7xl mx-auto px-6 h-16 flex flew-row items-center justify-between">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-full bg-black flex items-center justify-center">
          <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="5"/>
          </svg>
        </div>
        <RouterLink to="/" class="font-display text-2xl text-black-900">
          DyslexIA
        </RouterLink>
      </div>
      <div class="flex gap-6">
        <RouterLink
          v-for="l in links" :key="l.href" :to="l.href"
          class="text-sm font-light transition-colors duration-200 text-black-700 hover:text-black-900"
          active-class="text-black-600 font-medium"
        >
          {{ l.label }}
        </RouterLink>
      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
const links = [
  { href: '/', label: 'Présentation' },
  { href: '/test', label: 'Test' },
  { href: '/results', label: 'Résultats' },
]
</script>
