import type { NanoNet, NanoNetOptions } from '@/types/NanoNet'
import { MODES } from './config/constants'

export const createNanonet = (options?: NanoNetOptions): NanoNet => {
  // log options
  console.debug('constructing nanonet', options)

  // set defaults
  const DEFAULT_MODE = MODES.MULTILAYER_PERCEPTRON

  // create nanonet instance
  const nanonet: NanoNet = { mode: DEFAULT_MODE }

  // return the new instance
  return nanonet
}
