# nanonet

A lightweight, zero-dependency neural network class for JavaScript.

NanoNet implements a fully-connected feedforward neural network with sigmoid activation, trained by stochastic gradient descent (SGD) backpropagation. The whole library is a single ES module — `NanoNet/nanonet.js` — with a pre-minified copy at `NanoNet/nanonet.min.js`.

## Install Instructions

### via npm

- run install command from terminal

```bash
npm install nanonet
```

- Import the NanoNet class into your project, e.g.,

```javascript
import NanoNet from 'nanonet'
```

### via download

- *Download* or *Clone* repository and unzip.
- Import the NanoNet class into your project, e.g.,

```javascript
import NanoNet from './NanoNet/nanonet.js'
```

## How to Use NanoNet

### Initialise

Create a new neural network by passing a structure array to the constructor:

```javascript
// The structure can be any array of integers >= 1 and with length >= 2.
let structure = [8, 4, 2];
let neuralNetwork = new NanoNet(structure);
```

The array length defines the number of layers in the network, whilst the integer values define how many neurons should be initialised in each layer, e.g., a network with an input layer with 16 neurons and an output layer with 9 neurons would have a structure `[16,9]`. To add in 3 hidden layers, with 4, 5 and 6 neurons respectively, the structure becomes `[16,4,5,6,9]`.

**Important** \- The structure argument is required; calling `new NanoNet()` without one will throw.

On initialisation, weights are set randomly in the range [-1, 1) and biases start at 0.

### Methods

#### .feedForward(inputData)

```javascript
let inputData = [7.5, 0.40576, 8];
let fed = neuralNetwork.feedForward(inputData);
```

Given an array of input data, the feedForward method feeds the data forwards through the network and returns the NanoNet instance that was fed. *Data must be numeric.*

**Important** \- The length of the array must match the length of the input layer to the network. Hopefully this is intuitive as each piece of data corresponds to an input activation.

In order to retrieve the output activations use the **output** getter, e.g.,

```javascript
let output = fed.output;
```

#### .train(trainingData)

Given an array of training data, the network will be trained (i.e., will update its weights and biases using SGD style backpropagation).

Training data *must be numeric* and structured as follows:

```javascript
let trainingData = [
  // Each element in the array is a training instance.
  [
    // Each training instance is expected to hold two arrays, the first
    // holding the input data (see .feedForward)
    [7.5, 0.40576, 8],
    // and the second holding the expected output
    [0, 1],
  ], // ...
];
let trained = neuralNetwork.train(trainingData);
```

The train method returns the updated instance of the NanoNet class.

**Important** \- The length of the input and expected output arrays must match the input and output structure of the network.

### Properties

#### learningRate

The learning rate can be manually updated by reassigning the value of the property:

```javascript
let neuralNetwork = new NanoNet([2, 2, 2]);
neuralNetwork.learningRate = 0.02;
```

The default learning rate is 0.1.

### Getters

#### input

Returns an array containing the network's current input values.

#### output

Returns an array containing the network's current output activations.

## License

[MIT](http://opensource.org/licenses/MIT)
