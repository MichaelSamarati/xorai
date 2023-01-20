const { trainingData, testData } = require("./data");
const {
  createModel,
  convertToTensor,
  trainModel,
  testModel,
} = require("./model");

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

run();
