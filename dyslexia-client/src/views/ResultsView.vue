<template>
  <div>
    <h1>Résultats</h1>
    <div v-if="!store.data">
      <p>Aucune session enregistrée.</p>
      <RouterLink to="/test">Faire le test</RouterLink>
    </div>
    <div v-else>
      <div>
        <p> les Graphiques et resultats ici</p>
      </div>
      <button @click="store.clear()">Effacer et recommencer</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useSessionStore } from '@/stores/session'
const store = useSessionStore()
</script>
