/**
 * The shape of a network: one integer per layer giving the number of neurons
 * in that layer. Must contain at least two layers (input and output).
 */
export type Structure = readonly number[];

/**
 * A single training example: an input vector (matching the input layer size)
 * and the expected output vector (matching the output layer size).
 */
export type TrainingInstance = readonly [
  input: readonly number[],
  expected: readonly number[],
];

interface Layer {
  /** weights[j][k] connects neuron k in the previous layer to neuron j in this layer. */
  weights: number[][];
  biases: number[];
  /** Weighted inputs (z-values) from the most recent forward pass. */
  weightedInputs: number[];
  activations: number[];
}

const DEFAULT_STRUCTURE: Structure = [2, 2, 2];

/**
 * A fully-connected feedforward neural network with sigmoid activation,
 * trained by per-sample stochastic gradient descent backpropagation.
 */
export default class NanoNet {
  readonly structure: Structure;
  learningRate = 0.1;

  private readonly inputActivations: number[];
  private readonly layers: Layer[];

  constructor(structure: Structure = DEFAULT_STRUCTURE) {
    validateStructure(structure);
    this.structure = [...structure];
    this.inputActivations = new Array(structure[0]).fill(0);
    this.layers = [];
    for (let l = 1; l < structure.length; l++) {
      const size = structure[l];
      const previousSize = structure[l - 1];
      this.layers.push({
        weights: Array.from({ length: size }, () =>
          Array.from({ length: previousSize }, () => randomBetween(-1, 1)),
        ),
        biases: new Array(size).fill(0),
        weightedInputs: new Array(size).fill(0),
        activations: new Array(size).fill(0),
      });
    }
  }

  /** A copy of the network's current input values. */
  get input(): number[] {
    return [...this.inputActivations];
  }

  /** A copy of the network's current output activations. */
  get output(): number[] {
    return [...this.layers[this.layers.length - 1].activations];
  }

  /**
   * Feeds an input vector forwards through the network. Read the result from
   * the `output` getter.
   */
  feedForward(input: readonly number[]): this {
    validateVector(input, this.structure[0], 'input');
    for (let i = 0; i < input.length; i++) {
      this.inputActivations[i] = input[i];
    }
    this.feed();
    return this;
  }

  /**
   * Trains the network on an array of `[input, expected]` pairs, updating
   * weights and biases after each pair (per-sample SGD).
   */
  train(data: readonly TrainingInstance[]): this {
    const outputSize = this.structure[this.structure.length - 1];
    for (const [input, expected] of data) {
      validateVector(expected, outputSize, 'expected output');
      this.feedForward(input);
      this.propagateBackwards(expected);
    }
    return this;
  }

  private feed(): void {
    let previousActivations = this.inputActivations;
    for (const layer of this.layers) {
      for (let j = 0; j < layer.biases.length; j++) {
        const weightRow = layer.weights[j];
        let weightedInput = layer.biases[j];
        for (let k = 0; k < previousActivations.length; k++) {
          weightedInput += weightRow[k] * previousActivations[k];
        }
        layer.weightedInputs[j] = weightedInput;
        layer.activations[j] = NanoNet.sigmoid(weightedInput);
      }
      previousActivations = layer.activations;
    }
  }

  private propagateBackwards(expected: readonly number[]): void {
    const deltas = this.computeDeltas(expected);
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const layerDeltas = deltas[l];
      const previousActivations =
        l === 0 ? this.inputActivations : this.layers[l - 1].activations;
      for (let j = 0; j < layerDeltas.length; j++) {
        const step = layerDeltas[j] * this.learningRate;
        const weightRow = layer.weights[j];
        for (let k = 0; k < previousActivations.length; k++) {
          weightRow[k] -= step * previousActivations[k];
        }
        layer.biases[j] -= step;
      }
    }
  }

  /**
   * Computes the error term for every neuron, output layer first, working
   * backwards from the gradient of the squared-error loss.
   */
  private computeDeltas(expected: readonly number[]): number[][] {
    const deltas: number[][] = new Array(this.layers.length);
    const outputLayer = this.layers[this.layers.length - 1];
    deltas[this.layers.length - 1] = outputLayer.activations.map(
      (activation, j) =>
        (activation - expected[j]) *
        NanoNet.sigmoidDerivative(outputLayer.weightedInputs[j]),
    );
    for (let l = this.layers.length - 2; l >= 0; l--) {
      const followingLayer = this.layers[l + 1];
      const followingDeltas = deltas[l + 1];
      deltas[l] = this.layers[l].weightedInputs.map((weightedInput, j) => {
        let weightedDelta = 0;
        for (let k = 0; k < followingDeltas.length; k++) {
          weightedDelta += followingLayer.weights[k][j] * followingDeltas[k];
        }
        return weightedDelta * NanoNet.sigmoidDerivative(weightedInput);
      });
    }
    return deltas;
  }

  static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  static sigmoidDerivative(x: number): number {
    const sigX = NanoNet.sigmoid(x);
    return sigX * (1 - sigX);
  }
}

function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function validateStructure(structure: Structure): void {
  if (!Array.isArray(structure) || structure.length < 2) {
    throw new TypeError(
      'structure must be an array of at least two layer sizes.',
    );
  }
  for (const size of structure) {
    if (!Number.isInteger(size) || size < 1) {
      throw new RangeError(
        `Every layer size in structure must be an integer >= 1, got ${size}.`,
      );
    }
  }
}

function validateVector(
  vector: readonly number[],
  expectedLength: number,
  name: string,
): void {
  if (!Array.isArray(vector)) {
    throw new TypeError(`${name} must be an array of numbers.`);
  }
  if (vector.length !== expectedLength) {
    throw new RangeError(
      `${name} must have length ${expectedLength}, got ${vector.length}.`,
    );
  }
  for (const value of vector) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      throw new TypeError(`${name} must contain only numbers.`);
    }
  }
}
