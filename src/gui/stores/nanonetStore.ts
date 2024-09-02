import { ref, type Ref } from 'vue'
import { defineStore } from 'pinia'
import { createNanonet as createNewNanonet } from '@/nanonet'
import type { NanoNet } from '@/types/NanoNet'

export const useNanonetStore = defineStore('nanonetStore', () => {
  const creating = ref(false)
  const nanonet: Ref<NanoNet | null> = ref(null)

  const createNanonet = () => {
    creating.value = true
    nanonet.value = createNewNanonet()
    console.log('nanonet', nanonet)
  }

  return { nanonet, createNanonet, creating }
})
