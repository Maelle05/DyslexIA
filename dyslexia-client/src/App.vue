<template>
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
