import { DEFAULT_PERCEPTRON_INPUT } from '@/nanonet/config/constants'
import { getRandomFloat } from '@/nanonet/utilities/numbers'
import type { PerceptronOptions, PerceptronInput } from '@/types/Perceptron'
import { cloneDeep } from 'lodash'
import { Matrix } from 'ml-matrix'

export class Perceptron {
  // properties
  inputs: PerceptronInput[] = []

  // constructor
  constructor(config: PerceptronOptions) {
    if (config?.inputs?.length) this.inputs = config?.inputs || []
    if (config?.generate) this.generateInputs(config?.generate)
    if (config?.randomise) this.randomiseInputs(config?.max, config?.min)
  }

  // getters
  get biases() {
    return this.inputs.map(i => i.bias)
  }

  get weights() {
    return this.inputs.map(i => i.weight)
  }

  // methods
  activationFunction(v: number) {
    // TODO: implement this bit
    return v
  }

  getWeightedSum(testData: number[]) {
    // multiply 'inputs' vector by 'weights' then add 'biases' vector
    const inputVector = new Matrix([testData])
    const weightsVector = new Matrix([this.weights])
    const biasesVector = new Matrix([this.biases])
    const multiplied = Matrix.multiply(inputVector, weightsVector)
    const added = Matrix.add(multiplied, biasesVector)
    // debugger
    return added.sum()
    // return Matrix.add(Matrix.multiply(inputVector, weightsVector), biasesVector)
  }

  guess(testData: number[]) {
    if (this.inputs.length !== testData.length) throw 'invalid test data: invalid length'
    const ws = this.getWeightedSum(testData)
    return this.activationFunction(ws)
  }

  generateInputs(n = 1) {
    for (let i = 0; i < n; i++) this.inputs.push(cloneDeep(DEFAULT_PERCEPTRON_INPUT))
  }

  randomiseInputBiases(min = -1, max = 1) {
    this.inputs.forEach(i => (i.bias = getRandomFloat(min, max)))
  }

  randomiseInputs(min = -1, max = 1) {
    this.randomiseInputBiases(min, max)
    this.randomiseInputWeights(min, max)
  }

  randomise() {
    this.randomiseInputs()
  }

  randomiseInputWeights(min = -1, max = 1) {
    this.inputs.forEach(i => (i.weight = getRandomFloat(min, max)))
  }
}
