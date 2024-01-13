import nj, { NdArray } from "numjs";

function forwardPass(initialInput, hiddenLayerWeights, outputLayerWeights) {
  ///Make a forward pass through the network
  // Calculate the input to the hidden layer.
  const hidden_layer_in = nj.dot(initialInput, hiddenLayerWeights);
  // Calculate the hidden layer output.
  const hidden_layer_out = nj.sigmoid(hidden_layer_in);

  // Calculate the input to the output layer.
  const output_layer_in = nj.dot(hidden_layer_out, outputLayerWeights);
  // Calculate the output of the network.
  const output_layer_out = nj.sigmoid(output_layer_in);

  return { hidden_layer_out, output_layer_out };
}

function backwardPass(
  initialInput: NdArray,
  expectedOuput: number,
  learnRate: number,
  hiddenLayerOutput: NdArray,
  outputLayerOuput: number,
  outputLayerWeights: NdArray
) {
  /// Make a backward pass through the network

  const error = expectedOuput - outputLayerOuput;

  const outputErrorTerm = error * outputLayerOuput * (1 - outputLayerOuput);
  const deltaWeightsOutput = hiddenLayerOutput
    .multiply(learnRate)
    .multiply(outputErrorTerm);

  let hiddenErrorTerm = outputLayerWeights
    .multiply(outputErrorTerm)
    .multiply(hiddenLayerOutput)
    .multiply(1 - hiddenLayerOutput);

  initialInput = nj.array(initialInput).reshape(3, 1);
  hiddenErrorTerm = nj.array(hiddenErrorTerm).reshape(1, 2);
  const deltaWeightsHidden = learnRate * initialInput.multiply(hiddenErrorTerm);

  return { deltaWeightsHidden, deltaWeightsOutput };
}

// Create data to run through the network
const x = nj.array([0.5, 0.1, -0.2]);
const target = 0.6;
const learnrate = 0.5;
const weights_input_to_hidden = nj.array([
  [0.5, -0.6],
  [0.1, -0.2],
  [0.1, 0.7],
]);
const weights_hidden_to_output = nj.array([0.1, -0.3]);

// Forward pass
const { hidden_layer_out, output_layer_out } = forwardPass(
  x,
  weights_input_to_hidden,
  weights_hidden_to_output
);

// Backward pass
const { deltaWeightsHidden, deltaWeightsOutput } = backwardPass(
  x,
  target,
  learnrate,
  hidden_layer_out,
  output_layer_out,
  weights_hidden_to_output
);

console.log("Change in weights for hidden layer to output layer:");
console.log(deltaWeightsOutput);
console.log("Change in weights for input layer to hidden layer:");
console.log(deltaWeightsHidden);
