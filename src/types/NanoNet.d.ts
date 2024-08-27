import { MODES } from '@/nanonet/config/constants'

export type NanoNet = {
  mode: (typeof MODES)[keyof typeof MODES]
}

export type NanoNetOptions = {
  mode?: (typeof MODES)[keyof typeof MODES]
}
