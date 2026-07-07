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

/**
 * A computed (hidden or output) layer. All buffers are allocated once in the
 * constructor and reused for every feed/training sample, so the hot path
 * performs no allocation.
 *
 * Storage is deliberately plain number arrays rather than Float64Array: V8
 * keeps all-double arrays as contiguous unboxed doubles already, and
 * benchmarking showed flat Float64Array storage ~15% slower for these loops.
 */
interface Layer {
  /** weights[j][k] connects neuron k in the previous layer to neuron j in this layer. */
  readonly weights: number[][];
  readonly biases: number[];
  /** Weighted inputs (z-values) from the most recent forward pass. */
  readonly weightedInputs: number[];
  readonly activations: number[];
  /** Scratch buffer for backpropagation error terms. */
  readonly deltas: number[];
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
        deltas: new Array(size).fill(0),
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
      const { weights, biases, weightedInputs, activations } = layer;
      for (let j = 0; j < biases.length; j++) {
        const weightRow = weights[j];
        let weightedInput = biases[j];
        for (let k = 0; k < previousActivations.length; k++) {
          weightedInput += weightRow[k] * previousActivations[k];
        }
        weightedInputs[j] = weightedInput;
        activations[j] = NanoNet.sigmoid(weightedInput);
      }
      previousActivations = activations;
    }
  }

  private propagateBackwards(expected: readonly number[]): void {
    this.computeDeltas(expected);
    let previousActivations = this.inputActivations;
    for (const layer of this.layers) {
      const { weights, biases, deltas } = layer;
      for (let j = 0; j < deltas.length; j++) {
        const step = deltas[j] * this.learningRate;
        const weightRow = weights[j];
        for (let k = 0; k < previousActivations.length; k++) {
          weightRow[k] -= step * previousActivations[k];
        }
        biases[j] -= step;
      }
      previousActivations = layer.activations;
    }
  }

  /**
   * Computes the error term for every neuron into each layer's preallocated
   * `deltas` buffer, output layer first, working backwards from the gradient
   * of the squared-error loss.
   */
  private computeDeltas(expected: readonly number[]): void {
    const outputLayer = this.layers[this.layers.length - 1];
    for (let j = 0; j < outputLayer.deltas.length; j++) {
      outputLayer.deltas[j] =
        (outputLayer.activations[j] - expected[j]) *
        NanoNet.sigmoidDerivative(outputLayer.weightedInputs[j]);
    }
    for (let l = this.layers.length - 2; l >= 0; l--) {
      const { deltas, weightedInputs } = this.layers[l];
      const following = this.layers[l + 1];
      // Accumulate row by row over the following layer's weights so memory
      // access stays sequential instead of striding down a column.
      deltas.fill(0);
      for (let k = 0; k < following.deltas.length; k++) {
        const followingDelta = following.deltas[k];
        const weightRow = following.weights[k];
        for (let j = 0; j < deltas.length; j++) {
          deltas[j] += weightRow[j] * followingDelta;
        }
      }
      for (let j = 0; j < deltas.length; j++) {
        deltas[j] *= NanoNet.sigmoidDerivative(weightedInputs[j]);
      }
    }
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
