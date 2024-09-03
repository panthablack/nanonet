import type { NanoNetModel, NanoNetModelOptions } from '@/types/NanoNet'
import { MODES } from './config/constants'

export const createModel = (options?: NanoNetModelOptions): NanoNetModel => {
  // log options
  console.debug('Constructing New NanoNet Model', options)

  // set defaults
  const DEFAULT_MODE = MODES.MULTILAYER_PERCEPTRON

  // create nanonet instance
  const model: NanoNetModel = { mode: DEFAULT_MODE }

  // return the new instance
  return model
}
