const trainingData = [
  { input: [0, 0], label: [0] },
  { input: [0, 1], label: [1] },
  { input: [1, 0], label: [1] },
  { input: [1, 1], label: [0] },
];
var testData = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
  [0.5, 0.5],
  [0.1, 0.9],
  [Math.random(), Math.random()],
  [Math.random(), Math.random()],
  [Math.random(), Math.random()],
  [Math.random(), Math.random()],
];

module.exports = { trainingData, testData };
