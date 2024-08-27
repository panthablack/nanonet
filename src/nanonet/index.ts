import type { NanoNet, NanoNetOptions } from '@/types/NanoNet'

export const createNanonet = (options?: NanoNetOptions): NanoNet => {
  console.debug('constructing nanonet', options)
  const nanonet: NanoNet = {}
  return nanonet
}
