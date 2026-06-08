<template>
  <div>
    <section ref="heroRef">
      <h1>DyslexIA</h1>
      <p>Dépistage de la dyslexie par eye-tracking en temps réel.</p>
      <RouterLink to="/test">Commencer le test</RouterLink>
    </section>

    <section ref="featuresRef">
      <div v-for="f in features" :key="f.title" class="feature-card">
        <h3>{{ f.title }}</h3>
        <p>{{ f.desc }}</p>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

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
