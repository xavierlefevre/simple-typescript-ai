import { Perceptron } from "./perceptron";

// isWord positions:
// loved -> 1
// it -> 2
// amazing -> 3
// not -> 4
// good -> 5
// pure -> 6
// so -> 7
// liked -> 8
// happy -> 9
const reactionsToClassify = [
  { sentence: "loved it", inNumbers: [1, 1, 0, 0, 0, 0, 0, 0, 0], label: 1 },
  { sentence: "amazing", inNumbers: [0, 0, 1, 0, 0, 0, 0, 0, 0], label: 1 },
  { sentence: "not good", inNumbers: [0, 0, 0, 1, 1, 0, 0, 0, 0], label: 0 },
  { sentence: "not amazing", inNumbers: [0, 0, 1, 1, 0, 0, 0, 0, 0], label: 0 },
  {
    sentence: "pure amazingness",
    inNumbers: [0, 0, 1, 0, 0, 1, 0, 0, 0],
    label: 1,
  },
  { sentence: "so good", inNumbers: [0, 0, 0, 0, 1, 0, 1, 0, 0], label: 1 },
  { sentence: "good", inNumbers: [0, 0, 0, 0, 1, 0, 0, 0, 0], label: 1 },
  { sentence: "liked not", inNumbers: [0, 0, 0, 1, 0, 0, 0, 1, 0], label: 0 },
  { sentence: "loved not", inNumbers: [1, 0, 0, 1, 0, 0, 0, 0, 0], label: 0 },
  { sentence: "liked", inNumbers: [0, 0, 0, 0, 0, 0, 0, 1, 0], label: 1 },
];

const perceptronInstance = new Perceptron(9);

for (let i = 0; i < 50; i++) {
  reactionsToClassify.forEach((example) => {
    perceptronInstance.trainingPerceptron(example.inNumbers, example.label);
  });
}

const x = { sentence: "loved", inNumbers: [1, 0, 0, 0, 0, 0, 0, 0, 0] };
console.log(x, "is:", perceptronInstance.inferencePerceptron(x.inNumbers));

const y = { sentence: "not loved", inNumbers: [1, 0, 0, 1, 0, 0, 0, 0, 0] };
console.log(y, "is:", perceptronInstance.inferencePerceptron(y.inNumbers));

const z = { sentence: "not happy", inNumbers: [0, 0, 0, 1, 0, 0, 0, 0, 1] };
console.log(z, "is:", perceptronInstance.inferencePerceptron(z.inNumbers));
