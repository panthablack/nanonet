import assert from 'node:assert/strict';
import { test } from 'node:test';

import NanoNet, { type TrainingInstance } from '../src/nanonet.ts';

test('defaults to a [2,2,2] structure', () => {
  const net = new NanoNet();
  assert.deepEqual(net.structure, [2, 2, 2]);
  assert.equal(net.input.length, 2);
  assert.equal(net.output.length, 2);
  assert.equal(net.learningRate, 0.1);
});

test('rejects invalid structures', () => {
  assert.throws(() => new NanoNet([2]), TypeError);
  assert.throws(() => new NanoNet([2, 0]), RangeError);
  assert.throws(() => new NanoNet([2, 1.5, 2]), RangeError);
});

test('feedForward produces sigmoid activations and returns the instance', () => {
  const net = new NanoNet([3, 4, 2]);
  const fed = net.feedForward([7.5, 0.40576, 8]);
  assert.equal(fed, net);
  assert.deepEqual(net.input, [7.5, 0.40576, 8]);
  assert.equal(net.output.length, 2);
  for (const activation of net.output) {
    assert.ok(activation > 0 && activation < 1);
  }
});

test('feedForward rejects malformed input', () => {
  const net = new NanoNet([3, 2]);
  assert.throws(() => net.feedForward([1, 2]), RangeError);
  assert.throws(() => net.feedForward([1, 2, Number.NaN]), TypeError);
});

test('train rejects malformed expected output', () => {
  const net = new NanoNet([2, 2, 1]);
  assert.throws(() => net.train([[[0, 1], [0, 1]]]), RangeError);
});

test('getters return copies, not live internal state', () => {
  const net = new NanoNet([2, 2]);
  net.feedForward([1, 1]);
  const output = net.output;
  output[0] = 999;
  assert.notEqual(net.output[0], 999);
});

test('training reduces error on a deep network', () => {
  const net = new NanoNet([3, 5, 4, 2]);
  const data: TrainingInstance[] = [
    [[0, 0, 1], [1, 0]],
    [[1, 1, 0], [0, 1]],
    [[1, 0, 1], [1, 1]],
  ];
  const errorFor = (n: NanoNet) =>
    data.reduce((sum, [input, expected]) => {
      const output = n.feedForward(input).output;
      return (
        sum +
        expected.reduce((s, e, i) => s + (output[i] - e) ** 2, 0)
      );
    }, 0);

  const before = errorFor(net);
  for (let epoch = 0; epoch < 2000; epoch++) {
    net.train(data);
  }
  const after = errorFor(net);
  assert.ok(
    after < before / 4,
    `expected error to shrink substantially: before=${before}, after=${after}`,
  );
});

test('learns XOR', () => {
  const data: TrainingInstance[] = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
  ];
  // Random initialisation occasionally lands in a local minimum, so allow a
  // few fresh attempts before declaring failure.
  const attempts = 3;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const net = new NanoNet([2, 4, 1]);
    net.learningRate = 0.5;
    for (let epoch = 0; epoch < 5000; epoch++) {
      net.train(data);
    }
    const correct = data.every(([input, [expected]]) => {
      const [output] = net.feedForward(input).output;
      return Math.round(output) === expected;
    });
    if (correct) {
      return;
    }
  }
  assert.fail(`failed to learn XOR in ${attempts} attempts`);
});

test('sigmoid is numerically stable at extremes', () => {
  assert.equal(NanoNet.sigmoid(0), 0.5);
  assert.equal(NanoNet.sigmoid(1000), 1);
  assert.equal(NanoNet.sigmoid(-1000), 0);
  assert.equal(NanoNet.sigmoidDerivative(0), 0.25);
});
