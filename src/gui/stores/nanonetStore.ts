import { ref, type Ref } from 'vue'
import { defineStore } from 'pinia'
import { createModel as createNewModel } from '@/nanonet'
import type { NanoNetModelOptions, NanoNetModel } from '@/types/NanoNet'

export const useNanonetStore = defineStore('nanonetStore', () => {
  const models: Ref<NanoNetModel[]> = ref([])

  const isValidModel = (m: NanoNetModel) => !!m // TODO: validate model

  const createModel = (options: NanoNetModelOptions): NanoNetModel => {
    const model = createNewModel(options)
    if (isValidModel(model)) models.value.push(model)
    return model
  }

  return { createModel, models }
})
