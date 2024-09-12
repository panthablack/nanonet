export type PerceptronInput = {
  weight: number
  bias: number
}

export type PerceptronOptions = {
  inputs?: PerceptronInput[]
  randomise?: boolean
  min?: number
  max?: number
  generate?: number
}
