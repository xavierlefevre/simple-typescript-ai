const learningRate = 1;
const bias = 1;
const weights = [Math.random(), Math.random(), Math.random()]; // 3 weights: 2 neurons + 1 bias

function unitStepActivationFunction(value: number) {
  // Also called: Heaviside Step Function
  if (value > 0) {
    return 1;
  } else {
    return 0;
  }
}

function sigmoidFunction(value: number) {
  // Also called: Logistic Activation Function
  return 1 / (1 + Math.exp(-value));
}

function rectifiedLinearUnit(value: number) {
  return Math.max(0, value);
}

function inferencePerceptron(input1: number, input2: number) {
  let computedOutput =
    input1 * weights[0] + input2 * weights[1] + bias * weights[2]; // Weighted sum

  //     return rectifiedLinearUnit(computedOutput);
  //   return sigmoidFunction(computedOutput);
  return unitStepActivationFunction(computedOutput);
}

function trainingPerceptron(
  trainingInput1: number,
  trainingInput2: number,
  expectedOutput: number
) {
  // Also called: Linear Binary Classifier

  const computedOutput = inferencePerceptron(trainingInput1, trainingInput2);

  const error = expectedOutput - computedOutput;
  weights[0] += error * trainingInput1 * learningRate;
  weights[1] += error * trainingInput2 * learningRate;
  weights[2] += error * bias * learningRate;
}

// Training run
for (let i = 0; i < 50; i++) {
  trainingPerceptron(1, 1, 1); // True or true
  trainingPerceptron(1, 0, 1); // True or false
  trainingPerceptron(0, 1, 1); // False or true
  trainingPerceptron(0, 0, 0); // False or false
}

// Inference run
const x = 1;
const y = 0;
const computedOutput = inferencePerceptron(x, y);
console.log(x, "or", y, "is:", computedOutput);
console.log("weights:", weights);
