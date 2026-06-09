<template>
  <div>
    <section ref="heroRef" class="min-h-screen flex flex-col items-center justify-center text-center px-6">
      <h1 class="font-display text-black-900 text-5xl font-bold">DyslexIA</h1>
      <p class="mt-4 max-w-xl text-lg text-gray-600">Dépistage de la dyslexie par eye-tracking en temps réel.</p>
      <RouterLink
      to="/test"
      class="mt-8 rounded-full border px-6 py-3 text-black hover:bg-gray-50"
      >Commencer le test</RouterLink>
    </section>

    <section ref="featuresRef"
      class="grid gap-6 px-6 py-16 md:grid-cols-3"
    >
      <div v-for="f in features" :key="f.title"
        class="rounded-xl border p-6 shadow-sm"
      >
        <h3 class="text-xl font-semibold" >{{ f.title }}</h3>
        <p class="mt-2 text-gray-600">{{ f.desc }}</p>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const heroRef = ref<HTMLElement>()
const featuresRef = ref<HTMLElement>()

const features = [
  { title: 'Eye-Tracking', desc: 'Analyse saccades et fixations via MediaPipe' },
  { title: 'IA Prédiction', desc: 'Modèle entraîné sur patterns oculaires cliniques' },
  { title: 'Rapport Détaillé', desc: 'WPM, fixations, régressions, score global' },
  { title: 'Données Privées', desc: 'Traitement 100% local, aucune donnée transmise' },
  { title: 'Crédits', desc: 'Justine, Yoann, Maëlle, Manon' },
]

onMounted(async () => {
  const { gsap } = await import('gsap')
  const { ScrollTrigger } = await import('gsap/ScrollTrigger')
  gsap.registerPlugin(ScrollTrigger)

  gsap.from(heroRef.value!.children, {
    y: 40, opacity: 0, duration: 0.8, stagger: 0.15, ease: 'power3.out'
  })

  gsap.from('.feature-card', {
    scrollTrigger: { trigger: featuresRef.value, start: 'top 80%' },
    y: 30, opacity: 0, stagger: 0.1, duration: 0.6,
  })
})
</script>
