<template>
  <div class="max-w-4xl mx-auto px-4 py-8 mt-15">
    <h1>Résultats</h1>
    <div v-if="!store.data">
      <p>Aucune session enregistrée.</p>
      <RouterLink to="/test"
        class="mt-8 inline-block rounded-full border px-6 py-3 text-black hover:bg-gray-50"
      >Faire le test</RouterLink>
    </div>
    <div v-else>
      <div>
        <p> les Graphiques et resultats ici</p>
      </div>
      <button @click="store.clear()"
        class="mt-8 rounded-full border px-6 py-3 text-black hover:bg-gray-50"
      >Effacer et recommencer</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useSessionStore } from '@/stores/session'
const store = useSessionStore()
</script>
