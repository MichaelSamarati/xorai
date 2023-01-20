const tf = require("@tensorflow/tfjs");

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
  const epochs = 3000;
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

module.exports = { createModel, convertToTensor, trainModel, testModel };
