<template>
  <div
      class="fixed inset-0 z-[9999] flex items-center justify-center bg-white p-8 text-center lg:hidden"
    >
    <div>
      <h1 class="font-display text-black-900 text-5xl font-bold">DyslexIA</h1>
      <br>
      <h2 class="mb-4 text-2xl font-semibold">
        Application disponible uniquement sur ordinateur
      </h2>

      <p class="text-gray-600">
        Pour réaliser le test de dyslexie avec l'eye-tracking,
        veuillez utiliser un ordinateur équipé d'une webcam.
      </p>
    </div>
  </div>

  <div class="min-h-screen relative overflow-x-hidden bg-[#dde6ed]">

    <NavBar />

    <main class="relative z-10">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import NavBar from '@/components/NavBar.vue'
</script>

<style>

</style>
