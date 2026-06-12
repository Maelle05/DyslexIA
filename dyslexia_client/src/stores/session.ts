import { defineStore } from 'pinia'
import type { SessionData } from '@/types/index'

export const useSessionStore = defineStore('session', {
  state: () => ({
    data: null as SessionData | null,
    id: null as string | null
  }),
  actions: {
    save(data: SessionData) {
      this.data = data
    },
    clear() {
      this.data = null
      this.id = null
    },
    addId(name: string){
      this.id = name
    }
  },
})
