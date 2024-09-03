import { MODES } from '@/nanonet/config/constants'

export type NanoNetModel = {
  mode: (typeof MODES)[keyof typeof MODES]
}

export type NanoNetModelOptions = {
  mode?: (typeof MODES)[keyof typeof MODES]
}
