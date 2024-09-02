import { ref, type Ref } from 'vue'
import { defineStore } from 'pinia'
import { createNanonet } from '@/nanonet'
import type { NanoNet } from '@/types/NanoNet'

export const useNanonetStore = defineStore('nanonetStore', () => {
  const started = ref(false)
  const nanonet: Ref<NanoNet | null> = ref(null)

  const startNanonet = () => {
    started.value = true
    nanonet.value = createNanonet()
    console.log('nanonet', nanonet)
  }

  return { nanonet, startNanonet, started }
})
