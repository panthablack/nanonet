import type { PerceptronInput } from '@/types/Perceptron'

export const ACTIVATION_FUNCTIONS = {
  SIGMOID: 1,
  RELU: 2,
  TAN_H: 3,
  GELU: 4,
  SILU: 5,
  SWISH: 6,
} as const

export const DEFAULT_PERCEPTRON_INPUT: PerceptronInput = {
  weight: 1,
  bias: 1,
} as const

export const MODES = {
  MULTILAYER_PERCEPTRON: 1,
  CONVOLUTIONAL_NEURAL_NETWORK: 2,
  TRANSFORMER: 3,
} as const
