# nanonet
A lightweight neural network class for JavaScript.

## Install Instructions
- *Download* or *Clone* repository and unzip.
- Import the NanoNet class into your project, e.g.,

```javascript
import NanoNet from './nanonet'
```

## How to Use NanoNet

### Initialise

- After importing the NanoNet class into your project (see above), you can create a new neural network instance as follows:
```javascript
let neuralNetwork = new NanoNet();
```
This initialises a neural network with one input layer, one hidden layer and one output layer by default.

- To define the structure of the network, pass in an array to the class constructor as follows:
```javascript
// The structure can be any array of integers >= 1 and with length >= 2.  
let structure = [8,4,2];
let neuralNetwork = new NanoNet(structure);
```
The array length defines the number of layers in the network, whilst the integer values define how many neurons should be initialised in each layer, e.g., a network with an input layer with 16 neurons and an output layer with 9 neurons would have a structure [16,9].  To add in 3 hidden layers, with 4,5 and 6 neurons respectively, the structure becomes [16,4,5,6,9].

### Methods

#### .feedForward(inputData)
```javascript
let inputData = [7.5,0.40576,8];
let fed = neuralNetwork.feedForward(inputData);
```
Given an array of input data, the feedForward method feeds the data forwards through the network and returns the NanoNet instance that was fed.  *Data must be numeric.*

**Important** \- The length of the array must match the length of the input layer to the network.  Hopefully this is intuitive as each piece of data corresponds to an input activation.

#### .train(trainingData)

Given an array of training data, the network will be trained (i.e., will update its weights and biases using SGD style backpropagation).

Training data *must be numeric* and structured as follows:
```javascript
let trainingData = [
  // Each element in the array is a training instance.
  [
    // Each training instance is expected to hold two arrays, the first
    // holding the input data (see .feedForward)
    [7.5,0.40576,8],
    // and the second holding the expected output
    [0,1],
  ], // ...
];
let trained = neuralNetwork.train(trainingData);
```
The train method returns the updated instance of the NanoNet class.

**Important** \- The length of the input and expected output arrays must match the input and output structure of the network.

### Properties

#### .learningRate

The learning rate can be manually updated by reassigning the value of the property:
```javascript
let neuralNetwork = new NanoNet();
neuralNetwork.learningRate = 0.02;
```
The default learning rate is 0.1;

## License
[MIT](http://opensource.org/licenses/MIT)
