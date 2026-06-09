import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import TestView from '@/views/TestView.vue'
import ResultsView from '@/views/ResultsView.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/',        component: HomeView },
    { path: '/test',    component: TestView },
    { path: '/results', component: ResultsView },
  ],
})
