const tf = require("@tensorflow/tfjs");
const trainingData = [
  { input: [0, 0], label: [0] },
  { input: [0, 1], label: [1] },
  { input: [1, 0], label: [1] },
  { input: [1, 1], label: [0] },
];
const testData = [
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

async function run() {
  console.log("XOR-AI started");

  const model = createModel();
  console.log("Done Model Generation");

  const start = new Date();
  const tensorData = convertToTensor(trainingData);
  const { inputs, labels } = tensorData;
  await trainModel(model, inputs, labels);
  console.log(
    "Done Training in " + (new Date().getTime() - start.getTime()) + " ms"
  );

  const [normInputs, normPredictions] = await testModel(
    model,
    testData,
    tensorData
  );

  for (let i = 0; i < normPredictions.length; i++) {
    let in1 = i * 2;
    let in2 = in1 + 1;
    console.log(
      normInputs[in1].toFixed(2) +
        " " +
        normInputs[in2].toFixed(2) +
        " => " +
        normPredictions[i].toFixed(2)
    );
  }
}

function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [2],
      units: 4,
      activation: "sigmoid",
      useBias: true,
    })
  );
  model.add(
    tf.layers.dense({ units: 6, activation: "sigmoid", useBias: true })
  );
  model.add(
    tf.layers.dense({ units: 1, activation: "sigmoid", useBias: true })
  );
  return model;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map((d) => d.input);
    const labels = data.map((d) => d.label);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 2]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });
  const epochs = 2000;
  return await model.fit(inputs, labels, {
    epochs,
    shuffle: true,
  });
}

async function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [inputs, predictions] = tf.tidy(() => {
    const x = tf.tensor(inputData).reshape([inputData.length, 2]);
    const preds = model.predict(x);
    const unNormXs = x.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  return [inputs, predictions];
}

run();
