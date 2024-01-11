export class Perceptron {
  public learningRate: number;
  public bias: number;
  public weights: number[];

  constructor(numberOfInputs: number) {
    this.learningRate = 1;
    this.bias = 1;
    this.weights = [...Array(numberOfInputs + 1)].map(() => Math.random());
  }

  unitStepActivationFunction(value: number) {
    // Also called: Heaviside Step Function
    if (value > 0) {
      return 1;
    } else {
      return 0;
    }
  }

  sigmoidFunction(value: number) {
    // Also called: Logistic Activation Function
    return 1 / (1 + Math.exp(-value));
  }

  rectifiedLinearUnit(value: number) {
    return Math.max(0, value);
  }

  inferencePerceptron(input: number[]) {
    let computedOutput = input.reduce(
      (previousValue, currentValue, currentIndex) =>
        previousValue + currentValue * this.weights[currentIndex]
    );
    computedOutput += this.bias * this.weights[this.weights.length - 1];

    return this.rectifiedLinearUnit(computedOutput);
    // return this.sigmoidFunction(computedOutput);
    // return this.unitStepActivationFunction(computedOutput);
  }

  trainingPerceptron(trainingInput: number[], expectedOutput: number) {
    // Also called: Linear Binary Classifier

    const computedOutput = this.inferencePerceptron(trainingInput);

    const error = expectedOutput - computedOutput;
    trainingInput.forEach(
      (value, index) =>
        (this.weights[index] += error * value * this.learningRate)
    );
    this.weights[this.weights.length - 1] +=
      error * this.bias * this.learningRate;
  }
}
