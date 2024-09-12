import { computed, ref, type Ref } from 'vue'
import { defineStore } from 'pinia'
import { createPerceptron as createNewPerceptron } from '@/nanonet'
import type { PerceptronOptions } from '@/types/Perceptron'
import { Perceptron } from '@/nanonet/classes/Perceptron'

export const usePerceptronStoreStore = defineStore('perceptronStore', () => {
  const perceptrons: Ref<Perceptron[]> = ref([])

  const createPerceptron = (options: PerceptronOptions): Perceptron =>
    perceptrons.value[perceptrons.value.push(createNewPerceptron(options))]

  const latest = computed(() => perceptrons.value[perceptrons.value.length - 1] || null)

  return { createPerceptron, latest, perceptrons }
})
