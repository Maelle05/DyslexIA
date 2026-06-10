<template>
  <div>
    <section ref="heroRef" class="min-h-screen flex flex-col items-center justify-center text-center px-6">
      <h1 class="font-display text-black-900 text-8xl font-bold">DyslexIA</h1>
      <p class="mt-4 max-w-xl text-lg">Dépistage de la dyslexie par eye-tracking en temps réel</p>
      <div>
        <div class="rotate-[2deg] hover:scale-105 hover:-rotate-[5deg] transition-transform duration-300">
          <RouterLink
          to="/test"
          class="mt-8 text-lg font-bold bg-[#E86A64] py-2 px-3 relative inline-block font-medium
             before:absolute before:bottom-[0px] before:left-0
             before:h-[100%] before:w-full
             before:bg-[#299F9C] before:[-z-index:1]
             before:scale-x-0 before:origin-left
             before:transition-transform before:duration-300
             hover:before:scale-x-100"
          ><span class="relative z-10">Commencer le test</span></RouterLink>
        </div>
      </div>
    </section>

   <section ref="featuresRef" class="bg-[#E9E4DB] border-t border-[#CDC7B9]">
      <div
        v-for="(f, i) in features"
        :key="f.title"
        class="feature-block py-20 px-[100px] flex flex-row justify-between align-bottom gap-10 border-b border-[#CDC7B9]"
        :class="i % 2 === 0 ? 'mr-auto' : 'ml-auto text-right flex-row-reverse'"
      >
        <div class="max-w-[60vw]">
          <h3
            class="font-display text-black-900 text-3xl font-bold  py-3 px-6 inline-block mb-4"
            :style="{ backgroundColor: f.color }"
            :class="i % 2 === 0 ? '-rotate-[5deg]' : 'rotate-[3deg]'"
          >{{ f.title }}</h3>
          <p v-if="f.desc" class="mt-4 text-gray-600 font-bold whitespace-pre-line">{{ f.desc }}</p>
          <p v-if="f.text" class="mt-6 text-gray-600 text-left leading-relaxed whitespace-pre-line">{{ f.text }}</p>
        </div>
        <img
          v-if="f.img"
          :src="f.img"
          :alt="f.title"
          class="w-[300px] h-[350px] object-contain"
        />
      </div>
      <div class="w-full flex justify-center py-5">
        <div class="flex flex-col gap-1 items-center">
          <a href="https://github.com/Maelle05/DyslexIA" target="_blank" rel="noopener noreferrer" class="h-[50px] w-[50px] inline-block">
            <img src="https://img.icons8.com/?size=100&id=12599&format=png&color=000000" alt="" srcset="">
          </a>
          <div class="py-4 text-center text-sm text-gray-500 z-1">
            © {{ new Date().getFullYear() }}— All rights reserved
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const heroRef = ref<HTMLElement>()
const featuresRef = ref<HTMLElement>()

const features = [
  {
    title: 'Présentation',
    desc: "Qu'est-ce que la dyslexie ?",
    text: `La dyslexie est un trouble neurologique qui touche environ 5 à 10 % de la population mondiale, soit quelque 700 millions de personnes. Il s'agit d'un trouble d'apprentissage lié au langage. Ses symptômes varient d'une personne à l'autre. Elle affecte généralement la manière dont les personnes lisent et écrivent. \n
    Parmi l'ensemble des personnes ayant des difficultés à lire, écrire, parler et orthographier, environ 70 à 80 % souffrent d'un certain degré de dyslexie. La dyslexie est généralement définie comme un spectre de difficultés.
    Les méthodes actuelles de dépistage de la dyslexie reposent sur une série de tests de lecture, d'écriture et d'expression orale.
    L'inconvénient de ce système de tests est qu'ils sont assez coûteux et ne sont pas disponibles partout.`,
    img: '/imgs/humans-3.png',
    color: '#299F9C'
  },
  {
    title: 'Eye-Tracking',
    color: '#E86A64',
    desc: 'Analyse saccades, fixations et temp de lecture',
    text: "DyslexIA est une application capable d'utiliser les données d'oculométrie d'une personne lisant un texte devant sa webcam pour déterminer si cette personne est dyslexique ou non. \n \n Les données d'oculométrie (angles yaw/pitch de chaque iris) sont collectées à l'aide de MediaPipe FaceLandmarker.",
    img: '/imgs/humans-2.png'
  },
  { title: 'IA Prédiction',
    color: '#6E9960',
    desc: 'Modèle entraîné sur patterns oculaires cliniques',
    img: '/imgs/humans-4.png'
  },
  { title: 'Rapport Détaillé',
    color: '#299F9C',
    desc: 'fixations, régressions, score global, prediction',
    img: '/imgs/humans-1.png'
  },
  {
    title: 'Disclaimer',
    color: '#E86A64',
    text: "Cette application a été conçue à titre expérimental pour les utilisateurs francophones ; par conséquent, les résultats des prédictions, ainsi que la génération de texte et de questions, sont pour l'instant en français. \n L'eye-tracking donnera de meilleurs résultats avec des webcams de haute qualité et un bon éclairage. \n Cette application est une expérience et ne peut en aucun cas se substituer au diagnostic d'un professionnel. Si vous avez le moindre doute quant à une éventuelle dyslexie chez vous ou chez quelqu'un d'autre, nous vous recommandons de consulter un professionnel.",
    img: '/imgs/humans-5.png'
  },
  { title: 'Crédits',
    color: '#6E9960',
    desc: 'Justine Boulant \n Yoann Lemoine \n Maëlle Rabouan \n Manon Hell',
    text: 'Le Wagon Nantes #Batch-2275',
    img: '/imgs/humans-6.png'
  },
]

let ScrollTriggerInstance: any

onMounted(async () => {
  const { gsap } = await import('gsap')
  const { ScrollTrigger } = await import('gsap/ScrollTrigger')
  gsap.registerPlugin(ScrollTrigger)

  gsap.from(heroRef.value!.children, {
    y: 40, opacity: 0, duration: 0.8, stagger: 0.15, ease: 'power3.out'
  })

  const blocks = featuresRef.value!.querySelectorAll('.feature-block')

  blocks.forEach((el, i) => {
    gsap.from(el, {
      scrollTrigger: {
        trigger: el,
        start: 'top 85%',
      },
      x: i % 2 === 0 ? -60 : 60,
      opacity: 0,
      duration: 0.7,
      ease: 'power3.out',
    })
  })
})

onUnmounted(() => {
  ScrollTriggerInstance?.getAll().forEach((t: any) => t.kill())
})
</script>
