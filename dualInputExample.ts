import { Perceptron } from "./perceptron";

const perceptronInstance = new Perceptron(2);

// Training run
for (let i = 0; i < 50; i++) {
  perceptronInstance.trainingPerceptron([1, 1], 1); // True or true
  perceptronInstance.trainingPerceptron([1, 0], 1); // True or false
  perceptronInstance.trainingPerceptron([0, 1], 1); // False or true
  perceptronInstance.trainingPerceptron([0, 0], 0); // False or false
}

// Inference run
const x = 1;
const y = 0;
const computedOutput = perceptronInstance.inferencePerceptron([x, y]);
console.log(x, "or", y, "is:", computedOutput);
