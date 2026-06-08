import { defineStore } from 'pinia'
import type { SessionData } from '@/types'

export const useSessionStore = defineStore('session', {
  state: () => ({
    data: null as SessionData | null,
  }),
  actions: {
    save(data: SessionData) {
      this.data = data
    },
    clear() {
      this.data = null
    },
  },
})
