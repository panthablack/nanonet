export default class NanoNet {

    constructor(structure) {
        this.structure = structure;
        this.layers = [];
        this.learningRate = 0.1;
        this.layers[0] = {
            activations: [],
        };
        for (let j = 0; j < structure[0]; j++) {
            this.layers[0].activations.push(0);
        }
        for (let l = 1; l < structure.length; l++) {
            let layer = {
                activations: [],
                weights: [],
                biases: [],
            };
            for (let j = 0; j < structure[l]; j++) {
                layer.activations.push(0);
                layer.biases.push(0);
                layer.weights.push([]);
                for (let k = 0; k < structure[l - 1]; k++) {
                    layer.weights[j].push(NanoNet.random(-1,1));
                }
            }
            this.layers[l] = layer;
        }
    }

    get input() {
        return this.layers[0].activations;
    }

    get output() {
        return this.layers[this.layers.length - 1].activations;
    }

    train(data) {
        for (let i in data) {
            let datum = data[i];
            let input = datum[0];
            let expected = datum[1];
            this.feedForward(input);
            this.propagateBackwards(expected);
        }
        return this;
    }

    feedForward(input) {
        for (let i in input) {
            this.layers[0].activations[i] = input[i];
        }
        let fed = this.feed();
        return fed;
    }

    feed() {
        for (let l = 1; l < this.layers.length; l++) {
            let weighted = this.getWeightedInputs(l);
            let activated = this.applyActivationFunction(weighted);
            this.layers[l].activations = activated;
        }
        return this;
    }

    applyActivationFunction(arr) {
        let activated = []
        for (let i in arr) {
            activated[i] = NanoNet.sigmoid(arr[i]);         
        }
        return activated;
    }

    applyActivationFunctionDerivative(arr) {
        let differentiated = []
        for (let i in arr) {
            differentiated[i] = NanoNet.sigmoidDerivative(arr[i]);         
        }
        return differentiated;
    }

    propagateBackwards(expected) {
        let outputError = this.getOutputGradient(expected);
        let deltas = [];
        deltas[0] = this.getOutputDeltas(outputError);
        for (let l = 1; l < this.layers.length - 1; l++) {
            deltas[l] = this.getDeltas(deltas[l - 1],this.layers.length - 1 - l);
        }
        // make life easier by reversing the deltas
        deltas.reverse();
        this.updateWeightsAndBiases(deltas);
        return this;
    }

    updateWeightsAndBiases(deltas) {
        for (let l = 1; l < this.layers.length; l++) {
            let layer = this.layers[l];
            let previousLayer = this.layers[l - 1];
            let currentDeltas = deltas[l - 1];
            let currentWeights = layer.weights;
            let currentBiases = layer.biases;
            let inputActivations = previousLayer.activations;
            let deltaWeights = [];
            for (let j in currentDeltas) {
                deltaWeights[j] = [];
                for (let k in inputActivations) {
                    deltaWeights[j][k] = inputActivations[k] * currentDeltas[j];
                }
            }
            let deltaBiases = [];
            for (let j in currentDeltas) {
                deltaBiases[j] = currentDeltas[j];
            }
            for (let j in currentWeights) {
                let row = currentWeights[j];
                for (let k in row) {
                    currentWeights[j][k] -= (deltaWeights[j][k] * this.learningRate);
                }
            }

            for (let j in currentBiases) {
                currentBiases[j] -= (deltaBiases[j] * this.learningRate);
            }
        }
        return this;
    }

    getOutputGradient(expected) {
        let error = [];
        let output = this.output;
        for (let i in output) {
            error[i] = output[i] - expected[i];         
        }
        return error;
    }

    getOutputDeltas(outputError) {
        let deltas = [];
        let weighted = this.getWeightedInputs(this.layers.length - 1);
        let differentiated = this.applyActivationFunctionDerivative(weighted);
        for (let i in outputError) {
            deltas[i] = outputError[i] * differentiated[i];         
        }
        return deltas;
    }

    getWeightedInputs(l) {
        let layer = this.layers[l];
        let previousLayer = this.layers[l - 1];
        let inputActivations = previousLayer.activations;
        let weights = layer.weights;
        let biases = layer.biases;
        let weightedInputs = [];
        for (let j in biases) {
            let weightedActivations = [];
            for (let k in inputActivations) {
                weightedActivations[k] = weights[j][k] * inputActivations[k];
            }
            let summed = NanoNet.sum(weightedActivations);
            weightedInputs[j] = summed + biases[j];
        }
        return weightedInputs;
    }

    getWeightedDeltas(transposed, followingDeltas) {
        let weightedDeltas = [];
        for (let j in transposed) {
            let weightedDelta = [];
            for (let k in followingDeltas) {
                weightedDelta[k] = transposed[j][k] * followingDeltas[k];
            }
            let summed = NanoNet.sum(weightedDelta);
            weightedDeltas[j] = summed;
        }
        return weightedDeltas;
    }

    getDeltas(followingDeltas,l) {
        let followingLayer = this.layers[l + 1];
        let deltas = [];
        let weighted = this.getWeightedInputs(l);
        let differentiated = this.applyActivationFunctionDerivative(weighted);
        let transposed = this.transposeWeights(followingLayer.weights);
        let weightedDeltas = this.getWeightedDeltas(transposed, followingDeltas);
        for (let j in weightedDeltas) {
            deltas[j] = weightedDeltas[j] * differentiated[j];         
        }
        return deltas;
    }

    transposeWeights(weights) {
        let transposed = [];
        for (let j in weights) {
            let row = weights[j];
            for (let k in row) {
                if (!transposed[k]) {
                    transposed[k] = [];
                }
                transposed[k][j] = weights[j][k];
            }
        }
        return transposed;
    }

    // maths functions

    static random(x1 = 1, x2 = 0) {
        if (typeof x1 === 'number' && typeof x2 === 'number') {
            if (x1 !== x2) {
                let min = x1 < x2 ? x1 : x2;
                let max = x1 < x2 ? x2 : x1;
                return (Math.random() * (max - min)) + min;
            } else {
                throw 'Range must be non-zero.'
            }
        } else {
            throw 'Range min and max arguments must be valid numbers.'
        }
    }

    static sigmoid(x) {
        if (typeof x === 'number') {
            let esupx = Math.exp(x);
            let sig = esupx / (1 + esupx);
            if (sig < 0) {
                return 0;
            } else if (sig > 1) {
                return 1;
            } else if (Number.isNaN(sig)) {
                return 0.5;
            } else {
                return sig;
            }
        } else {
            throw 'Argument must be a number.'
        }
    }

    static sigmoidDerivative(x) {
        if (typeof x === 'number') {
            let sigX = NanoNet.sigmoid(x);
            return sigX * (1 - sigX);
        } else {
            throw 'Argument must be a number.'
        }
    }

    static sum(arr) {
        if (Array.isArray(arr) && arr.length) {
            let sum = 0;
            for (let i = 0; i < arr.length; i++) {
                let n = arr[i];
                if (typeof n === 'number' && !Number.isNaN(n)) {
                    sum += n;
                } else {
                    throw 'Sum function can only sum numbers.'
                }
            }
            return sum;
        } else {
            throw 'Sum function can only accept arrays as arguments.'
        }
    }


    // utility functions

    halt(message = null) {
        if (message) {
            console.log(message);
        }
        throw 'halted';
    }

}

